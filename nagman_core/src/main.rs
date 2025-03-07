pub mod paging;

use paging::unsafe_api::wrap_vram::AllVram;

fn main() {
    let vram_size: usize = 1024; // 1MB
    let vram = AllVram::new(vram_size).expect("CUDAメモリ確保に失敗しました");

    println!("5秒待機");
    std::thread::sleep(std::time::Duration::from_secs(5));

    println!("メモリ確保成功:");
    println!("  アドレス: {:?}", vram.ptr());
    println!("  サイズ: {} bytes", vram.size());
    println!("  要求サイズ: {} bytes", vram_size);
}
