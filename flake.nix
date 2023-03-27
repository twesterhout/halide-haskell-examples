{
  description = "twesterhout/halide-haskell-examples: Examples implemented in halide-haskell and numba";

  nixConfig = {
    extra-experimental-features = "nix-command flakes";
    extra-substituters = "https://halide-haskell.cachix.org";
    extra-trusted-public-keys = "halide-haskell.cachix.org-1:cFPqtShCsH4aNjn2q4PHb39Omtd/FWRhrkTBcSrtNKQ=";
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    nixGL = {
      url = "github:guibou/nixGL";
      inputs.flake-utils.follows = "flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    hdf5-hs = {
      url = "github:twesterhout/hdf5-hs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    halide-haskell = {
      url = "github:twesterhout/halide-haskell";
      inputs.nixGL.follows = "nixGL";
      inputs.flake-utils.follows = "flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs: inputs.flake-utils.lib.eachDefaultSystem (system:
    with builtins;
    let
      inherit (inputs.nixpkgs) lib;
      pkgs = import inputs.nixpkgs { inherit system; };

      haskellPackagesOverride = ps: args:
        let
          ghcAttr = "ghc" + (concatStringsSep "" (splitVersion ps.ghc.version));
        in
        ps.override
          {
            overrides = self: super: {
              halide-haskell = inputs.halide-haskell.outputs.packages.${system}.${ghcAttr}.halide-haskell;
              hdf5-hs = inputs.hdf5-hs.outputs.packages.${system}.${ghcAttr}.hdf5-hs;
              # example-by-danis-badrtdinov =
              #   (self.callCabal2nix "example-by-danis-badrtdinov" ./from_danis_badrtdinov { });
              example-by-hugo-strand =
                (self.callCabal2nix "example-by-hugo-strand" ./from_hugo_strand { });
            };
          };
      haskellPackages = haskellPackagesOverride pkgs.haskellPackages { };
    in
    {
      packages = {
        default = haskellPackages.example-by-hugo-strand;
      };
      devShells = {
        default =
          haskellPackages.shellFor {
            packages = ps: with ps; [
              example-by-hugo-strand
            ];
            withHoogle = true;
            nativeBuildInputs = with pkgs; with haskellPackages; [
              cabal-install
              (python3.withPackages (ps: with ps; [ numpy scipy numba h5py ]))
              # LSP
              haskell-language-server
              nil
              nodePackages.pyright
              # Formatters
              python3Packages.black
              fourmolu
              cabal-fmt
              nixpkgs-fmt
              # Previewing markdown files
              python3Packages.grip
            ];
          };
      };
      formatter = pkgs.nixpkgs-fmt;
    }
  );
}
