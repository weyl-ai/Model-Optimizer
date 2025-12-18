{
  description = "nvidia-modelopt // dev";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        {
          config,
          self',
          inputs',
          system,
          lib,
          ...
        }:
        let
          # Apply CUDA overlay at the nixpkgs level
          pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [
              (final: prev: {
                # Override default CUDA packages to use version 13
                cudaPackages = final.cudaPackages_13;
              })
            ];
          };

          inherit (lib) optionals optionalString;

          # Load workspace
          workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          packageOverrides =
            final: prev:
            let
              addCudaDeps =
                pkg: deps:
                pkg.overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];
                  buildInputs = (old.buildInputs or [ ]) ++ deps;

                  # add `rpath` for NVIDIA libraries that put their .so files in python site-packages
                  postFixup = (old.postFixup or "") + ''
                    for dep in ${lib.concatStringsSep " " (map (d: "${d}") deps)}; do
                      if [ -d "$dep/lib" ]; then
                        addAutoPatchelfSearchPath "$dep/lib"
                      fi
                      site_packages=$(find "$dep" -type d -name "site-packages" 2>/dev/null | head -n1)
                      if [ -n "$site_packages" ]; then
                        nvidia_libs=$(find "$site_packages/nvidia" -name "*.so*" -type f 2>/dev/null | xargs -r dirname | sort -u)
                        for libdir in $nvidia_libs; do
                          addAutoPatchelfSearchPath "$libdir"
                        done
                      fi
                    done
                  '';
                });
            in
            {
              torch = addCudaDeps prev.torch [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.nccl

                final.nvidia-cublas-cu12
                final.nvidia-cuda-cupti-cu12
                final.nvidia-cuda-nvrtc-cu12
                final.nvidia-cuda-runtime-cu12
                final.nvidia-cudnn-cu12
                final.nvidia-cufft-cu12
                final.nvidia-cufile-cu12
                final.nvidia-curand-cu12
                final.nvidia-cusolver-cu12
                final.nvidia-cusparse-cu12
                final.nvidia-cusparselt-cu12
                final.nvidia-nccl-cu12
                final.nvidia-nvshmem-cu12
              ];

              nvidia-cufile-cu12 = addCudaDeps prev.nvidia-cufile-cu12 [ pkgs.rdma-core ];

              nvidia-nvshmem-cu12 = addCudaDeps prev.nvidia-nvshmem-cu12 [
                pkgs.libfabric
                pkgs.mpi
                pkgs.pmix
                pkgs.rdma-core
                pkgs.ucx
              ];

              nvidia-cusparse-cu12 = addCudaDeps prev.nvidia-cusparse-cu12 [
                final.nvidia-nvjitlink-cu12
              ];

              nvidia-cusolver-cu12 = addCudaDeps prev.nvidia-cusolver-cu12 [
                final.nvidia-cublas-cu12
                final.nvidia-cusparse-cu12
                final.nvidia-nvjitlink-cu12
              ];

              tensorrt = prev.tensorrt.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  final.setuptools
                  pkgs.autoPatchelfHook
                ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgs.cudaPackages.tensorrt
                  pkgs.cudaPackages.cudatoolkit
                ];
              });

              tensorrt-cu13 = prev.tensorrt-cu13.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  final.setuptools
                  pkgs.autoPatchelfHook
                ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgs.cudaPackages.tensorrt
                  pkgs.cudaPackages.cudatoolkit
                ];
              });

              tensorrt-cu13-bindings = prev.tensorrt-cu13-bindings.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  final.setuptools
                  pkgs.autoPatchelfHook
                ];
                buildInputs = (old.buildInputs or [ ]) ++ [
                  pkgs.cudaPackages.tensorrt
                  pkgs.cudaPackages.cudatoolkit
                ];
              });

              torchvision = prev.torchvision.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];

                buildInputs = (old.buildInputs or [ ]) ++ [
                  final.torch
                  pkgs.cudaPackages.cudatoolkit
                ];

                postFixup = (old.postFixup or "") + ''
                  echo "Starting custom postFixup for torchvision" >&2

                  # Try to add torch paths
                  if [ -d "${final.torch}/lib" ]; then
                    echo "Adding torch lib path: ${final.torch}/lib" >&2
                    addAutoPatchelfSearchPath "${final.torch}/lib"
                  fi

                  # Look for torch site-packages
                  torch_site=$(find "${final.torch}" -type d -name "site-packages" 2>/dev/null | head -n1)
                  if [ -n "$torch_site" ]; then
                    echo "Found torch site-packages at: $torch_site" >&2
                    torch_libs=$(find "$torch_site/torch" -name "*.so*" -type f 2>/dev/null | xargs -r dirname | sort -u)
                    for libdir in $torch_libs; do
                      echo "Adding torch lib dir: $libdir" >&2
                      addAutoPatchelfSearchPath "$libdir"
                    done
                  fi

                  echo "Finished custom postFixup for torchvision" >&2
                '';
              });
            };

          pythonSet =
            (pkgs.callPackage inputs.pyproject-nix.build.packages {
              python = pkgs.python312;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  inputs.pyproject-build-systems.overlays.default
                  overlay
                  packageOverrides
                ]
              );

          pythonEnv = pythonSet.mkVirtualEnv "nvidia-modelopt" (workspace.deps.default or { });

          basePackages = [
            pkgs.uv
            pkgs.ruff
            pkgs.git
            pkgs.cudaPackages.tensorrt
            pkgs.cudaPackages.nsight_systems
            pkgs.cudaPackages.nsight_compute
          ]
          ++ optionals pkgs.stdenv.isLinux [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.tensorrt
            pkgs.libfabric
            pkgs.mpi
            pkgs.rdma-core
            pkgs.stdenv.cc.cc.lib
            pkgs.ucx
          ];

          # LD_LIBRARY_PATH for Linux
          ldLibraryPath = optionalString pkgs.stdenv.isLinux (
            lib.makeLibraryPath (
              [
                pkgs.cudaPackages.cudatoolkit.lib
                pkgs.cudaPackages.cudnn
                pkgs.libfabric
                pkgs.mpi
                pkgs.rdma-core
                pkgs.stdenv.cc.cc.lib
                pkgs.ucx
              ]
              ++ pkgs.pythonManylinuxPackages.manylinux1
            )
            + ":/run/opengl-driver/lib"
          );
        in
        {
          devShells = {
            default = pkgs.mkShell {
              name = "nvidia-modelopt-dev";

              packages = basePackages ++ [ pythonEnv ];

              CUDA_HOME = optionalString pkgs.stdenv.isLinux "${pkgs.cudaPackages.cudatoolkit}";
              TRITON_LIBCUDA_PATH = optionalString pkgs.stdenv.isLinux "/run/opengl-driver/lib";
              LD_LIBRARY_PATH = ldLibraryPath;

              shellHook = ''
                echo "nvidia-modelopt development environment"
                echo "Python: ${pythonEnv}/bin/python"
                ${optionalString pkgs.stdenv.isLinux ''
                  echo "CUDA: ${pkgs.cudaPackages.cudatoolkit.version}"
                  echo "TensorRT: ${pkgs.cudaPackages.tensorrt.version}"
                ''}
                source "${pythonEnv}/bin/activate"
                echo "Environment ready"
              '';
            };
          };

          packages = {
            default = pythonEnv;

            # n.b. expose individual python packages for debugging
            inherit (pythonSet)
              nvidia-cusparselt-cu12
              nvidia-cusparse-cu12
              nvidia-cufile-cu12
              nvidia-nvshmem-cu12
              nvidia-cusolver-cu12
              torch
              torchvision
              ;
          };
        };
    };
}
