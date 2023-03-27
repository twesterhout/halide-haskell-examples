{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Control.Monad (forM_, void)
import Data.HDF5 qualified as H5
import Data.Some
import Debug.TimeStats qualified as TimeStats (measureM, printTimeStats)
import Language.Halide
import Prelude hiding (div, mod)

nextColumn :: Expr Int32 -> Func 'ParamTy 4 Double -> IO (Func 'FuncTy 4 Double)
nextColumn n b' = do
  [k, m, i, j] <- mapM mkVar ["k", "n", "i", "j"]
  b <- constantExterior 0 b'
  let upper =
        cast @Double (2 * (n + k) `mod` 2 - 1)
          * (2 * cast @Double k + 1)
          / (2 * cast @Double n + 3)
          * b ! (n + 1, k, i, j)
  let lower =
        -(2 * cast @Double n + 1) / (2 * cast @Double k + 3) * b ! (k + 1, n, i, j)
          + (2 * cast @Double n + 1) / (2 * cast @Double k - 1) * b ! (k - 1, n, i, j)
          + b ! (k, n - 1, i, j)
  f <- define "next" (k, m, i, j) $ bool (k `lt` n) upper lower
  ki <- mkVar "ki"
  void $ split TailShiftInwards k (k, ki) 8 f >>= vectorize ki
  pure f

spectralConvolutionMatrix :: IO (Double -> Ptr (HalideBuffer 3 Double) -> Ptr (HalideBuffer 4 Double) -> IO ())
spectralConvolutionMatrix = do
  [k, n, i, j] <- mapM mkVar ["k", "n", "i", "j"]
  mkColumn0 <- compile $ \s a' -> do
    a <- constantExterior 0 a'
    define "column0" (k, n, i, j) $
      bool (k `eq` 0) (s * a ! (k, i, j)) 0
        + a ! (k - 1, i, j) / (2 * cast @Double k - 1)
        - a ! (k + 1, i, j) / (2 * cast @Double k + 3)
  mkColumn1 <- compile $ \s b' -> do
    b <- constantExterior 0 b'
    define "column1" (k, n, i, j) $
      bool (k `eq` 0) (-s * b ! (k + 1, 0, i, j) / 3) $
        b ! (k - 1, 0, i, j) / (2 * cast @Double k - 1)
          - s * b ! (k, 0, i, j)
          - b ! (k + 1, 0, i, j) / (2 * cast @Double k + 3)
  mkNext <- compile nextColumn
  let convolution !s !a !mat = do
        size <- getBufferExtent mat 1
        withCropped mat 1 0 1 $ mkColumn0 s a
        withCropped mat 1 1 1 $ mkColumn1 s mat
        let go !n'
              | n' < size = do
                  withCropped mat 1 n' 1 $ mkNext (fromIntegral (n' - 1)) mat
                  go (n' + 1)
              | otherwise = pure ()
        go 2
  pure convolution

main :: IO ()
main = do
  (g :: [[[Double]]]) <-
    H5.withFile "data_in.h5" H5.ReadOnly $ \h -> do
      H5.open h "G" >>= \case
        (Some dset@(H5.Dataset _)) -> H5.readDataset dset
        _ -> error "expected 'G' to be a dataset"
  let n0 = length g
      n1 = length (head g)
      n2 = length (head (head g))
  kernel <- spectralConvolutionMatrix
  out <-
    allocaCpuBuffer @4 @Double [n0, n0, n1, n2] $ \out' ->
      withHalideBuffer @3 @Double g $ \g' -> do
        forM_ [(0 :: Int) .. 99] $ \_ -> do
          TimeStats.measureM "kernel" $ kernel 1 g' out'
        peekToList out'
  H5.withFile "data_out_new.h5" H5.WriteTruncate $ \h ->
    void $ H5.createDataset h "mat" out

  TimeStats.printTimeStats
