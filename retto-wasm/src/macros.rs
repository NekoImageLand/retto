/// helper
#[macro_export]
macro_rules! _str_to_bytes {
    ($s:expr) => {{
        const S: &str = $s;
        const LEN: usize = S.len();
        const BYTES: [u8; LEN] = {
            let bytes = S.as_bytes();
            let mut array = [0u8; LEN];
            let mut i = 0;
            while i < LEN {
                array[i] = bytes[i];
                i += 1;
            }
            array
        };
        BYTES
    }};
}

/// em_macros.h
#[macro_export]
macro_rules! em_import {
    ($js_name:ident, $c_name:ident, ($($param:ident: $ty:ty),* $(,)?), $ret:ty) => {
        #[link(wasm_import_module = "env")]
        unsafe extern "C" {
            #[link_name = stringify!($js_name)]
            fn $c_name($($param: $ty),*) -> $ret;
        }
    };
}

#[macro_export]
macro_rules! em_js_deps {
    ($tag:ident, $deps:literal) => {
        paste::paste! {
            #[used]
            #[unsafe(no_mangle)]
            #[unsafe(link_section = "em_lib_deps")]
            pub static [<__em_lib_deps_ $tag>]: [u8; concat!($deps).len()] =
                $crate::_str_to_bytes!(concat!($deps));
        }
    };
}

/// em_js.h
#[macro_export]
macro_rules! _em_js_internal {
    ($ret:ty, $c_name:ident, $js_name:ident, ($($param_name:ident: $param_type:ty),*), $code:expr) => {
        paste::paste! {
            em_import!(
                $js_name,
                $c_name,
                ($($param_name: $param_type),*),
                $ret
            );

            #[used]
            #[allow(non_upper_case_globals)]
            #[unsafe(export_name = concat!("__em_js_ref_", stringify!($c_name)))]
            pub static [<__em_js_ref_ $c_name>]: unsafe extern "C" fn($($param_type),*) -> $ret =
                $c_name;

            #[used]
            #[linkage = "external"]
            #[allow(non_upper_case_globals)]
            #[unsafe(link_section = "em_js")]
            #[unsafe(export_name = concat!("__em_js__", stringify!($js_name)))]
            pub static [<__em_js__ $js_name>]: [u8; concat!(
                concat!("(", stringify!($($param_name),*), ")" ),
                "<::>",
                $code,
                "\0"
            ).len()] = _str_to_bytes!(concat!(
                concat!("(", stringify!($($param_name),*), ")" ),
                "<::>",
                $code,
                "\0"
            ));
        }
    };
}

/// Since Rust's functional macros cannot capture space semantics using `stringify!($($code)+)`,
/// currently only highly inlined JS functions can be entered
#[macro_export]
macro_rules! em_js {
    ($ret:ty, $name:ident, ($($param_name:ident: $param_type:ty),*), $($code:tt)+) => {
        _em_js_internal!(
            $ret,
            $name,
            $name,
            ($($param_name: $param_type),*),
            stringify!($($code)+)
        );
    };
}

/// Currently has the same limitations as [em_js]. See [em_js] for details.
#[macro_export]
macro_rules! em_async_js {
    ($ret:ty, $name:ident, ($($param_name:ident: $param_type:ty),*), $($code:tt)+) => {
        paste::paste! {
            _em_js_internal!(
                $ret,
                $name,
                [<__asyncjs__ $name>],
                ($($param_name: $param_type),*),
                concat!("{ return Asyncify.handleAsync(async () => ", stringify!($($code)+), "); }")
            );
        }
    };
}
