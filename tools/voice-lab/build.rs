fn main() {
    println!("cargo:rerun-if-changed=n64tts/n64tts.c");
    println!("cargo:rerun-if-changed=n64tts/n64tts.h");
    cc::Build::new().file("n64tts/n64tts.c").opt_level(3).compile("n64tts");
}
