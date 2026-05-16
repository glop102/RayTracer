final: prev: {
  raytracer = final.stdenv.mkDerivation {
    pname = "raytracer";
    version = "1.0";

    src = ./src_cpp;

    buildInputs = [final.libpng];
    makeFlags = [
      "DESTDIR=$(out)"
    ];

    meta = {
      description = "Does some raytracing with a custom concept implementation.";
    };
  };
}
