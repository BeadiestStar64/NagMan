//! # Unsafe API
//!
//! ## 注意⚠️
//! このモジュールはunsafeなAPIを提供しています。
//! このAPIを使う際は、呼び出し側で十分な注意を払ってください。
//! また、Rust公式のunsafeドキュメントを参照して、unsafeなコードを書く際のガイドラインを理解してください。
//!
//! ## 概要
//! このモジュールは、NagManの内部で使われるunsafeなAPIを提供します。
//! 具体的には、以下の機能を**unsafe**として提供します。
//!  - GPUのメモリ操作

pub mod wrap_vram {
    //! # VRAM管理用スマートポインター
    //!
    //! AllVram構造体は、CUDAで確保したVRAMの0番目のアドレスとバイト単位の確保サイズを保持します。
    //! 内部リソースは、Dropトレイトで自動的に解放されます。

    use super::cuda_connector::{allocate_vram, free_vram, get_memory_info, CudaError};
    use std::ffi::c_void;

    pub struct AllVram {
        ptr: *mut c_void,
        actual_size: usize,     // 実際に確保されたサイズ（推定値）
    }

    impl AllVram {
        //! 指定されたバイト数のVRAMを確保し、AllVramインスタンスを生成します。
        //!
        //! # 引数
        //! - size: 確保するバイト数
        //!
        //! # 戻り値
        //! 成功時はAllVramのインスタンスを返し、失敗時はCUDAのエラーコードを返します。
        pub fn new(size: usize) -> Result<Self, CudaError> {
            // メモリ確保前の空きメモリ量を取得
            let (free_before, _) = get_memory_info().unwrap_or((0, 0));

            match allocate_vram(size) {
                Ok(ptr) => {
                    // メモリ確保後の空きメモリ量を取得して実際に確保されたサイズを推定
                    let (free_after, _) = get_memory_info().unwrap_or((free_before, 0));
                    let actual_size = if free_before > free_after {
                        free_before - free_after
                    } else {
                        // 推定できない場合は要求サイズをそのまま使用
                        size
                    };

                    Ok(AllVram {
                        ptr,
                        actual_size
                    })
                },
                Err(err) => Err(err),
            }
        }

        // 確保されたVRAMの先頭ポインタを返します。
        pub fn ptr(&self) -> *mut c_void {
            self.ptr
        }

        // 実際に確保されたVRAMサイズ（バイト単位）の推定値を返します。
        pub fn size(&self) -> usize {
            self.actual_size
        }
    }

    impl Drop for AllVram {
        fn drop(&mut self) {
            // VRAM解放に失敗した場合はエラーコードを出力します。
            if let Err(err) = free_vram(self.ptr) {
                eprintln!("VRAMの解放に失敗しました。エラーコード: {}", err);
            }
        }
    }
}

pub mod cuda_connector {
    //! # CUDA VRAM操作のunsafe API
    //!
    //! このモジュールは、CUDAのFFIを利用してGPUのVRAM確保と解放を行うunsafeなAPIを提供します。
    //! 本APIを利用する際は、エラー処理とリソース管理に十分注意してください。

    use std::ffi::c_void;
    use std::ptr;

    pub type CudaError = i32;
    // CUDAランタイムの成功を示す定数（実際の値はCUDAのドキュメントを参照してください）
    const CUDA_SUCCESS: CudaError = 0;

    // FFIバインディング: CUDAのcudaMallocおよびcudaFreeの宣言
    #[link(name = "cudart")]
    unsafe extern "C" {
        // CUDAのVRAM確保関数
        pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
        // CUDAのVRAM解放関数
        pub fn cudaFree(ptr: *mut c_void) -> CudaError;
        // CUDAのエラーメッセージ取得関数
        pub fn cudaGetErrorString(error: CudaError) -> *const i8;
        // CUDAデバイス情報取得関数
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    }

    /// CUDAのVRAMを確保するラッパー関数
    ///
    /// 成功時は確保されたメモリアドレスを返し、失敗時はCUDAのエラーコードを返します。
    pub fn allocate_vram(size: usize) -> Result<*mut c_void, CudaError> {
        // GPUメモリの使用状況を確認
        let (free, total) = get_memory_info().unwrap_or((0, 0));
        println!("CUDAメモリ情報:");
        println!("  総メモリ: {:.2} GB", bytes_to_gb(total));
        println!("  空きメモリ: {:.2} GB", bytes_to_gb(free));
        println!("  使用中: {:.2} GB", bytes_to_gb(total - free));
        println!("  要求サイズ: {:.2} GB", bytes_to_gb(size));

        if size > free {
            println!("警告: 要求サイズが空きメモリを超えています！");
        }

        let mut ptr: *mut c_void = ptr::null_mut();
        println!("cudaMalloc呼び出し中...");

        // unsafeブロック内でFFI関数を呼び出します。
        let err = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, size) };

        if err != CUDA_SUCCESS {
            // エラー発生時はエラーコードとエラーメッセージを表示
            let error_msg = get_error_string(err);
            println!("VRAM確保失敗: エラーコード={}, メッセージ={}", err, error_msg);
            Err(err)
        } else {
            // 確保後のメモリ使用状況を再取得
            if let Ok((free_after, _)) = get_memory_info() {
                let estimated_allocated = free - free_after;
                println!("VRAM確保成功:");
                println!("  アドレス: {:?}", ptr);
                println!("  要求サイズ: {} bytes ({:.2} GB)", size, bytes_to_gb(size));
                println!("  推定確保サイズ: {} bytes ({:.2} GB)", estimated_allocated, bytes_to_gb(estimated_allocated));
                if estimated_allocated != size {
                    println!("  警告: 推定確保サイズと要求サイズが一致しません");
                    println!("  差異: {} bytes ({:.2} MB)",
                        (size as isize - estimated_allocated as isize).abs(),
                        bytes_to_mb((size as isize - estimated_allocated as isize).unsigned_abs()));
                }
            }

            Ok(ptr)
        }
    }

    /// CUDAで確保したVRAMを解放するラッパー関数
    ///
    /// 成功時はOk(())、失敗時はCUDAのエラーコードを返します。
    pub fn free_vram(ptr: *mut c_void) -> Result<(), CudaError> {
        println!("VRAMを解放中: {:?}", ptr);
        let (free_before, _) = get_memory_info().unwrap_or((0, 0));

        let err = unsafe { cudaFree(ptr) };

        if err != CUDA_SUCCESS {
            let error_msg = get_error_string(err);
            println!("VRAM解放失敗: エラーコード={}, メッセージ={}", err, error_msg);
            Err(err)
        } else {
            // 解放後のメモリ使用状況を確認
            if let Ok((free_after, _)) = get_memory_info() {
                let freed_size = free_after - free_before;
                println!("VRAM解放成功:");
                println!("  解放サイズ（推定）: {} bytes ({:.2} MB)", freed_size, bytes_to_mb(freed_size));
            }

            Ok(())
        }
    }

    /// CUDAのエラーコードからエラーメッセージを取得
    fn get_error_string(error: CudaError) -> String {
        unsafe {
            let ptr = cudaGetErrorString(error);
            if ptr.is_null() {
                return "不明なエラー".to_string();
            }

            // C文字列をRustの文字列に変換
            let c_str = std::ffi::CStr::from_ptr(ptr as *const i8);
            match c_str.to_str() {
                Ok(s) => s.to_string(),
                Err(_) => "エラーメッセージのデコードに失敗".to_string(),
            }
        }
    }

    /// CUDAデバイスのメモリ使用状況を取得
    pub fn get_memory_info() -> Result<(usize, usize), CudaError> {
        let mut free: usize = 0;
        let mut total: usize = 0;

        let err = unsafe { cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize) };

        if err != CUDA_SUCCESS {
            Err(err)
        } else {
            Ok((free, total))
        }
    }

    // ユーティリティ関数：バイトをGB単位に変換
    fn bytes_to_gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    // ユーティリティ関数：バイトをMB単位に変換
    fn bytes_to_mb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{cuda_connector::allocate_vram, wrap_vram::AllVram, *};

    #[test]
    fn test_cuda_malloc() {
        println!("[AllVram使用]CUDAのVRAM確保テスト");

        // サイズ計算を明示的にキャストして行う
        let gb: usize = 2;
        let size: usize = gb * 1024 * 1024 * 1024;

        println!("要求サイズ: {} bytes ({} GB)", size, gb);

        // Alloc構造体を使用して安全にメモリを確保
        match AllVram::new(size) {
            Ok(vram) => {
                // サイズの単位変換を行い、GB、MB単位でも表示
                let size_mb = vram.size() as f64 / (1024.0 * 1024.0);
                let size_gb = size_mb / 1024.0;

                println!("メモリ確保成功:");
                println!("  アドレス: {:?}", vram.ptr());
                println!("  サイズ: {} bytes ({:.2} MB / {:.4} GB)", vram.size(), size_mb, size_gb);
                println!("  要求サイズとの差: {} bytes", size as isize - vram.size() as isize);

                // ここでvramを使った処理を行う
                // AllVram構造体はスコープを抜けると自動的にVRAMを解放するので、
                // 明示的なfree_vramは不要
            },
            Err(code) => {
                // エラー処理
                println!("VRAM確保失敗: エラーコード={}", code);
                panic!("CUDAメモリ確保に失敗しました");
            },
        }
        println!("5秒待機開始");
        std::thread::sleep(std::time::Duration::from_secs(5));
        println!("VRAM確保テスト完了");
    }

    #[test]
    fn test_allocate_free_vram_directly() {
        println!("[直接API使用]CUDAのVRAM確保テスト");

        // 1MBのメモリ確保を試みる
        let size: usize = 1024 * 1024;
        println!("要求サイズ: {} bytes (1 MB)", size);

        // 直接allocate_vramとfree_vramを呼び出すテスト
        match allocate_vram(size) {
            Ok(ptr) => {
                println!("メモリ確保成功: アドレス={:?}", ptr);
                println!("5秒待機開始");
                std::thread::sleep(std::time::Duration::from_secs(5));
                println!("VRAM確保テスト完了");
                // 必ずメモリを解放する
                match cuda_connector::free_vram(ptr) {
                    Ok(_) => println!("メモリ解放成功"),
                    Err(e) => panic!("メモリ解放失敗: エラーコード={}", e),
                }
            },
            Err(code) => {
                println!("VRAM確保失敗: エラーコード={}", code);
                panic!("CUDAメモリ確保に失敗しました");
            },
        }
    }
}
