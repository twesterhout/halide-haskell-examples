{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}

module Main where

import Control.Monad (forM_, void)
import Data.List (transpose)
import Data.Vector.Storable.Mutable qualified as MV
import Debug.TimeStats qualified as TimeStats (measureM, printTimeStats)
import Language.Halide hiding (StageSchedule (atomic))
import System.IO
import Prelude hiding (and, div, mod)

reciprocalVectors :: [[Double]]
reciprocalVectors = [[1.57119, 0.90712, 0.00000], [0.00000, 1.81425, 0.00000], [0.00000, 0.00000, 0.296097]]

latticeVectors :: [[Double]]
latticeVectors = [[3.999, 0.000, 0.00000], [-1.999, 3.463, 0.00000], [0.000, 0.000, 21.2200]]

sia :: Double
sia = 0.00071

loadInput :: FilePath -> (Ptr (HalideBuffer 5 Double) -> IO a) -> IO a
loadInput filename action = do
  j_r <- MV.new $ 3 * 3 * 11 * 11 * 1
  withFile filename ReadMode $ \h -> do
    let isInteresting ('#' : _) = True
        isInteresting _ = False
    contents <- hGetContents h
    forM_ (fmap (drop 1 . words) . filter isInteresting . lines $ contents) $ \case
      [read -> a1, read -> a2, read -> i, read -> j, read -> k, value] ->
        MV.write j_r (a1 + 3 * (a2 + 3 * ((i + 5) + 11 * ((j + 5) + 11 * k)))) $ read @Double value
      l -> error $ "invalid line: " <> show l
  MV.unsafeWith j_r $ \cpuPtr ->
    bufferFromPtrShape cpuPtr [3, 3, 11, 11, 1] action

reciprocalSpaceGrid ::
  Expr Int32 ->
  Expr Int32 ->
  Expr Int32 ->
  IO (Func 'FuncTy 1 (Expr Double, Expr Double))
reciprocalSpaceGrid d0 d1 d2 = do
  n <- mkVar "n"
  define "grid" n $
    let x = n `div` (d1 * d2)
        y = n `mod` (d1 * d2) `div` d2
        z = n `mod` (d1 * d2) `mod` d2
        relativeCoordinates =
          [ cast @Double x / cast @Double d0
          , cast @Double y / cast @Double d1
          , cast @Double z / cast @Double d2
          ]
        qx = sum $ zipWith (*) (mkExpr <$> head (transpose reciprocalVectors)) relativeCoordinates
        qy = sum $ zipWith (*) (mkExpr <$> transpose reciprocalVectors !! 1) relativeCoordinates
     in (qx, qy)

traceInverse :: Func t 3 (Expr Double, Expr Double) -> IO (Func 'FuncTy 1 (Expr Double))
traceInverse n_q = do
  n <- mkVar "n"
  let [[r00, r01, r02], [_, r11, r12], [_, _, r22]] =
        [[fst (n_q ! (n, mkExpr i, mkExpr j)) | j <- [0 .. 2]] | i <- [0 .. 2]]
  let [[_, i01, i02], [_, _, i12], [_, _, _]] =
        [[snd (n_q ! (n, mkExpr i, mkExpr j)) | j <- [0 .. 2]] | i <- [0 .. 2]]
  let numerator =
        r00 * r11
          + r00 * r22
          + r11 * r22
          - (r01 * r01 + i01 * i01)
          - (r02 * r02 + i02 * i02)
          - (r12 * r12 + i12 * i12)
  let trivial =
        r00 * r11 * r22
          - (r01 * r01 + i01 * i01) * r22
          - (r02 * r02 + i02 * i02) * r11
          - (r12 * r12 + i12 * i12) * r00
  let fancy = 2 * (i02 * i12 * r01 - i01 * i12 * r02 + i01 * i02 * r12 + r01 * r02 * r12)
  define "w" n $ numerator / (trivial + fancy)

integrate ::
  Expr Int32 ->
  Expr Int32 ->
  Expr Int32 ->
  Func 'ParamTy 5 (Expr Double) ->
  IO (Func 'FuncTy 0 (Expr Double))
integrate d0 d1 d2 (buffer "j_r" -> j_r) = do
  grid <- reciprocalSpaceGrid d0 d1 d2
  a1 <- mkVar "a1"
  a2 <- mkVar "a2"
  -- 2 dimensional reduction to sum over the real space
  (rx, ry) <- toRVars =<< mkRDom @2 "r" (-5, -5) (10, 10)

  -- Fourier transform for q = (0, 0)
  j_q0 <- define "J_q0" (a1, a2) 0
  update j_q0 (a1, a2) $ j_q0 ! (a1, a2) + j_r ! (a1, a2, rx + 5, ry + 5, 0)

  -- Compute diagonal shift
  j_q0_symm <-
    define "J_q0_symm" (a1, a2) $ 0.5 * (j_q0 ! (a1, a2) + j_q0 ! (a2, a1))
  diag <- do
    a2' <- mkRVar "a2'" 0 3
    diag <- define "diag" a1 (-mkExpr sia)
    update diag a1 $ diag ! a1 + j_q0_symm ! (a1, a2')
    pure diag

  -- General Fourier transform
  n <- mkVar "n"
  j_q <- define "J_q" (n, a1, a2) (0, 0)
  update j_q (n, a1, a2) $
    let (qx, qy) = grid ! n
        position =
          zipWith
            (+)
            ((cast @Double rx *) <$> fmap mkExpr (head latticeVectors))
            ((cast @Double ry *) <$> fmap mkExpr (latticeVectors !! 1))
        phase = sum $ zipWith (*) position [qx, qy, 0]
        (re, im) = j_q ! (n, a1, a2)
        t = j_r ! (a1, a2, rx + 5, ry + 5, 0)
        c_re = cos phase
        c_im = negate . sin $ phase
     in (re + c_re * t, im + c_im * t)

  j_q_symm <-
    define "J_q_symm" (n, a1, a2) $
      let (re1, im1) = j_q ! (n, a1, a2)
          (re2, im2) = j_q ! (n, a2, a1)
       in (0.5 * (re1 + re2), 0.5 * (im1 - im2))
  n_q <-
    define "N_q" (n, a1, a2) $
      let (j_q_re, j_q_im) = j_q_symm ! (n, a1, a2)
       in (j_q_re - bool (eq a1 a2) (diag ! a1) 0, j_q_im)
  -- Compute the contribution to the integral
  w <- traceInverse n_q

  -- Compute the integral itself
  r <- mkRVar "r_n" 0 (d0 * d1 * d2)
  integral <- define "integral" () 0
  update integral () $ integral ! () + w ! r / 11604.525

  ri <- mkRVar "r_n_i" 0 8
  void $
    getUpdateStage 0 integral
      >>= split TailGuardWithIf r (r, ri) 8
      >>= reorder [ri, r]
      >>= atomic False
      >>= parallel r

  a <- mkVar "a"
  void $ fuse (a1, a2) a j_q0 >>= vectorize a >>= computeRoot
  void $ getUpdateStage 0 j_q0 >>= fuse (a1, a2) a >>= vectorize a
  void $ computeRoot diag

  ni <- mkVar "ni"
  void $ split TailShiftInwards n (n, ni) 8 w >>= vectorize ni
  void $ computeAt w =<< getLoopLevel integral r

  void $ fuse (a1, a2) a j_q >>= vectorize a
  void $ getUpdateStage 0 j_q >>= fuse (a1, a2) a >>= unroll a >>= reorder [n, a, rx, ry]

  pure integral

main :: IO ()
main = do
  integrateKernel <- TimeStats.measureM "compile" $ compile integrate
  loadInput "J_loc.dat" $ \j_r -> do
    let d0 = 256
        d1 = 256
        d2 = 1
    allocaCpuBuffer [] $ \out -> do
      TimeStats.measureM "kernel" $
        integrateKernel d0 d1 d2 j_r out
      t <- ((2 / 3) * fromIntegral (d0 * d1 * d2) /) <$> peekScalar out
      putStrLn $ "Tc: " <> show t
  TimeStats.printTimeStats
