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
        inherit (self.legacyPackages.${system}) raytracer;
        default = self.legacyPackages.${system}.raytracer;
      });
      apps = forAllSystems (system: {
        raytracer = {
          type = "app";
          program = "${self.legacyPackages.${system}.raytracer}/bin/raytracer";
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
          }
        )
      );
    };
}
