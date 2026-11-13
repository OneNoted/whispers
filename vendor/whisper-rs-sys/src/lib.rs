#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// bindgen's bitfield accessors use transmute; the lint is unknown before 1.88.
#![allow(unknown_lints, unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
