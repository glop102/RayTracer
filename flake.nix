{
  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    };
  };
  outputs =
    { self, nixpkgs, ... }@flakeInputs:
    let
      forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;
    in
    {
      inherit nixpkgs;
      overlays = {
        default = import ./overlay.nix;
      };
      legacyPackages = forAllSystems (
        system: nixpkgs.legacyPackages.${system}.appendOverlays (builtins.attrValues self.overlays)
      );
      formatter = forAllSystems (system: self.legacyPackages.${system}.nixfmt-tree);
      packages = forAllSystems (system: {
        inherit (self.legacyPackages.${system}) raytracer raytracer_vk;
        default = self.legacyPackages.${system}.raytracer;
      });
      apps = forAllSystems (system: {
        raytracer = {
          type = "app";
          program = "${self.legacyPackages.${system}.raytracer}/bin/raytracer";
        };
        raytracer_vk = {
          type = "app";
          program = "${self.legacyPackages.${system}.raytracer_vk}/bin/raytracer_vk";
        };
        default = self.apps.${system}.raytracer;
      });
      devShells = forAllSystems (
        system:
        (
          let
            pkgs = self.legacyPackages.${system};
            pythonPackages = pkgs.python3Packages;
          in
          {
            default = pkgs.mkShell {
              name = "ray-tracer dev shell";
              buildInputs =
                (with pkgs; [
                  ffmpeg
                  gnumake
                  gcc
                ]);
              shellHook = ''
                export PS1='\n(dev) \[\033[1;32m\][\[\e]0;\u@\h: \w\a\]\u@\h:\w]\$\[\033[0m\] '
              '';
            };
            vulkan = pkgs.mkShell {
              name = "vulkan dev shell";
              buildInputs = with pkgs; [
                pkg-config
                shaderc
                vulkan-headers
                vulkan-loader
                vulkan-validation-layers
                vulkan-tools
                vk-bootstrap
                vulkan-memory-allocator
                glfw
                glm
                tinygltf
                gnumake
                gcc
              ];
              shellHook = ''
                export PS1='\n(vk-dev) \[\033[1;32m\][\[\e]0;\u@\h: \w\a\]\u@\h:\w]\$\[\033[0m\] '
                export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
                # Mesa's Dozen (D3D12) driver returns VK_ERROR_INCOMPATIBLE_DRIVER but then
                # crashes in vkDestroyInstance. Pin VK_ICD_FILENAMES to the system GPU drivers
                # in /run/opengl-driver (NixOS) to exclude the dzn ICD from the Nix store.
                if [ -d /run/opengl-driver/share/vulkan/icd.d ]; then
                  export VK_ICD_FILENAMES=$(find /run/opengl-driver/share/vulkan/icd.d -name "*.json" | tr '\n' ':')
                fi
              '';
            };
          }
        )
      );
    };
}
