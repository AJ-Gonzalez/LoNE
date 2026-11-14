class LoneAne < Formula
  include Language::Python::Virtualenv

  desc "Run LFM3 and LFM2.5 on the Apple Neural Engine"
  homepage "https://github.com/AJ-Gonzalez/LoNE"
  url "https://github.com/AJ-Gonzalez/LoNE/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "c7c8c2d3e0e856820476768fa0ab5d02151c4cba9a4249057b2e9a3e751755af"
  license "Apache-2.0"
  version "0.2.0"

  depends_on "python@3.12"
  depends_on :macos => :sequoia

  def install
    venv = virtualenv_create(libexec, "python3.12", system_site_packages: false)
    python = Formula["python@3.12"].opt_bin/"python3.12"
    # The Homebrew helper installs --no-deps (resources are expected); this tap
    # resolves the runtime dependencies from PyPI instead.
    system python, "-m", "pip", "--python=#{venv.root}/bin/python", "install", "--no-cache-dir", "pip"
    system venv.root/"bin/pip", "install", "--no-cache-dir", buildpath.to_s
    bin.install_symlink venv.root/"bin/lone"
  end

  def caveats
    <<~EOS
      The runtime is installed (run/serve/verify). Conversion extras are not:
        #{opt_libexec}/bin/pip install "torch==2.7.0" "scikit-learn>=1.4,<1.6" "transformers>=5.0"
    EOS
  end

  test do
    assert_match "Usage", shell_output("#{bin}/lone --help")
    system libexec/"bin/python", "-c",
           "import coremltools, numpy, fastapi, uvicorn, tokenizers, safetensors, jinja2, typer"
  end
end
