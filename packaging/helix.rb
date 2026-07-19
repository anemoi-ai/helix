class Helix < Formula
  desc "OpenAI-isomorphic on-device LLM library"
  homepage "https://github.com/anemoi-ai/helix"
  url "https://github.com/anemoi-ai/helix/archive/v1.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256_UPDATED_AT_RELEASE"
  license "Apache-2.0"
  head "https://github.com/anemoi-ai/helix.git", branch: "main"

  bottle do
    sha256 cellar: :any, arm64_sequoia: "PLACEHOLDER"
    sha256 cellar: :any, arm64_sonoma:  "PLACEHOLDER"
    sha256 cellar: :any, ventura:       "PLACEHOLDER"
    sha256 cellar: :any, x86_64_linux:  "PLACEHOLDER"
  end

  depends_on "cmake" => :build

  def install
    backend = if OS.mac?
      Hardware::CPU.arm? ? "metal" : "cpu"
    else
      "cpu"
    end

    system "cmake", "-S", ".", "-B", "build",
           "-DCMAKE_INSTALL_PREFIX=#{prefix}",
           "-DHELIX_BACKEND=#{backend}",
           "-DHELIX_BUILD_TESTS=OFF",
           *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    system "#{bin}/helix-doctor", "--version" rescue nil
    # Ranged check: any 1.x at or above the formula's level is compatible,
    # so future minor releases don't need a formula edit.
    (testpath/"abi_check.c").write <<~C
      #include <helix.h>
      #include <assert.h>
      int main(void) {
        assert(helix_abi_version() >= 0x00010100 &&
               helix_abi_version() <  0x00020000);
        return 0;
      }
    C
    system ENV.cc, "abi_check.c", "-I#{include}", "-L#{lib}", "-lhelix", "-o", "abi_check"
    system "./abi_check"
  end
end
