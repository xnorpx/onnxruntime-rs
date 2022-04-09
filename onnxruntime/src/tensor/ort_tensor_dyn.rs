//! Module containing tensor with memory owned by Rust

use std::{ffi, fmt::Debug};

use ndarray::{ArrayBase, Data};
use tracing::{debug, error};

use onnxruntime_sys as sys;

use crate::{
    error::{assert_not_null_pointer, call_ort, status_to_result},
    g_ort,
    memory::MemoryInfo,
    session::Session,
    OrtError, Result, TensorElementDataType, TypeToTensorElementDataType,
};

/// Owned tensors, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
#[derive(Debug, Default)]
pub struct OrtTensorsDyn<'t> {
    pub(crate) inner: Vec<OrtTensorDyn<'t>>,
}

impl<'t> OrtTensorsDyn<'t> {
    /// Appends an element to the back of a collection.
    pub fn push<'m, T>(&mut self, tensor: &T, session: &'m Session) -> Result<()>
    where
        'm: 't, // 'm outlives 't
        T: super::AsOrtTensorDyn<'t>,
    {
        self.inner.push(tensor.as_ort_tensor_dyn(session)?);
        Ok(())
    }
}

/// Owned tensor, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
///
/// This tensor bounds the ONNX Runtime to `ndarray`; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
#[derive(Debug)]
pub struct OrtTensorDyn<'t> {
    pub(crate) c_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
    _array: &'t (),
}

impl<'t> OrtTensorDyn<'t> {
    pub(crate) fn from_array<'m, T, D>(
        memory_info: &'m MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        array: &ArrayBase<T, D>,
    ) -> Result<OrtTensorDyn<'t>>
    where
        'm: 't, // 'm outlives 't
        T: Data,
        T::Elem: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        Ok(OrtTensorDyn {
            c_ptr: Self::from_array_get_ptr(memory_info, allocator_ptr, array)?,
            shape: array.shape().to_vec(),
            _array: &(),
        })
    }

    pub(crate) fn from_array_get_ptr<'m, T, D>(
        memory_info: &'m MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        array: &ArrayBase<T, D>,
    ) -> Result<*mut sys::OrtValue>
    where
        'm: 't, // 'm outlives 't
        T: Data,
        T::Elem: TypeToTensorElementDataType + Debug + Clone,
        D: ndarray::Dimension,
    {
        // where onnxruntime will write the tensor data to
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;

        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();

        match T::Elem::tensor_element_data_type() {
            TensorElementDataType::I8
            | TensorElementDataType::I16
            | TensorElementDataType::I32
            | TensorElementDataType::I64
            | TensorElementDataType::U8
            | TensorElementDataType::U16
            | TensorElementDataType::U32
            | TensorElementDataType::U64
            | TensorElementDataType::F32
            | TensorElementDataType::F64 => {
                // Primitive data is already suitably laid out in memory; provide it to
                // onnxruntime as is
                //
                // note: ReleaseValue won't release `tensor_values_ptr`.
                let tensor_values_ptr: *mut std::ffi::c_void =
                    array.as_ptr() as *mut std::ffi::c_void;
                assert_not_null_pointer(tensor_values_ptr, "TensorValues")?;

                unsafe {
                    call_ort(|ort| {
                        ort.CreateTensorWithDataAsOrtValue.unwrap()(
                            memory_info.ptr,
                            tensor_values_ptr,
                            array.len() * std::mem::size_of::<T::Elem>(),
                            shape_ptr,
                            shape_len,
                            T::Elem::tensor_element_data_type().into(),
                            tensor_ptr_ptr,
                        )
                    })
                }
                .map_err(OrtError::CreateTensorWithData)?;
                assert_not_null_pointer(tensor_ptr, "Tensor")?;

                let mut is_tensor = 0;
                let status = unsafe { g_ort().IsTensor.unwrap()(tensor_ptr, &mut is_tensor) };
                status_to_result(status).map_err(OrtError::IsTensor)?;
            }
            TensorElementDataType::String => {
                // create tensor without data -- data is filled in later
                unsafe {
                    call_ort(|ort| {
                        ort.CreateTensorAsOrtValue.unwrap()(
                            allocator_ptr,
                            shape_ptr,
                            shape_len,
                            T::Elem::tensor_element_data_type().into(),
                            tensor_ptr_ptr,
                        )
                    })
                }
                .map_err(OrtError::CreateTensor)?;

                // create null-terminated copies of each string, as per `FillStringTensor` docs
                let null_terminated_copies: Vec<ffi::CString> = array
                    .iter()
                    .map(|elt| {
                        let slice = elt
                            .try_utf8_bytes()
                            .expect("String data type must provide utf8 bytes");
                        ffi::CString::new(slice)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(OrtError::CStringNulError)?;

                let string_pointers = null_terminated_copies
                    .iter()
                    .map(|cstring| cstring.as_ptr())
                    .collect::<Vec<_>>();

                unsafe {
                    call_ort(|ort| {
                        ort.FillStringTensor.unwrap()(
                            tensor_ptr,
                            string_pointers.as_ptr(),
                            string_pointers.len(),
                        )
                    })
                }
                .map_err(OrtError::FillStringTensor)?;
            }
        }

        assert_not_null_pointer(tensor_ptr, "Tensor")?;

        Ok(tensor_ptr)
    }
}

impl<'t> Drop for OrtTensorDyn<'t> {
    #[tracing::instrument]
    fn drop(&mut self) {
        // We need to let the C part free
        debug!("Dropping Tensor.");
        if self.c_ptr.is_null() {
            error!("Null pointer, not calling free.");
        } else {
            unsafe { g_ort().ReleaseValue.unwrap()(self.c_ptr) }
        }

        self.c_ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::call_ort, AllocatorType, MemType};
    use ndarray::{arr0, arr1, arr2, arr3};
    use std::ptr;
    use test_log::test;

    #[test]
    fn orttensor_from_array_0d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr0::<i32>(123);
        let tensor = OrtTensorDyn::from_array(&memory_info, ptr::null_mut(), &array).unwrap();
        let expected_shape: &[usize] = &[];
        assert_eq!(tensor.shape, expected_shape);
    }

    #[test]
    fn orttensor_from_array_1d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = OrtTensorDyn::from_array(&memory_info, ptr::null_mut(), &array).unwrap();
        let expected_shape: &[usize] = &[6];
        assert_eq!(tensor.shape, expected_shape);
    }

    #[test]
    fn orttensor_from_array_2d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = OrtTensorDyn::from_array(&memory_info, ptr::null_mut(), &array).unwrap();
        assert_eq!(tensor.shape, &[2, 6]);
    }

    #[test]
    fn orttensor_from_array_3d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = OrtTensorDyn::from_array(&memory_info, ptr::null_mut(), &array).unwrap();
        assert_eq!(tensor.shape, &[3, 2, 6]);
    }

    #[test]
    fn orttensor_from_array_1d_string() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[
            String::from("foo"),
            String::from("bar"),
            String::from("baz"),
        ]);
        let tensor =
            OrtTensorDyn::from_array(&memory_info, ort_default_allocator(), &array).unwrap();
        assert_eq!(tensor.shape, &[3]);
    }

    #[test]
    fn orttensor_from_array_3d_str() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [["1", "2", "3"], ["4", "5", "6"]],
            [["7", "8", "9"], ["10", "11", "12"]],
        ]);
        let tensor =
            OrtTensorDyn::from_array(&memory_info, ort_default_allocator(), &array).unwrap();
        assert_eq!(tensor.shape, &[2, 2, 3]);
    }

    fn ort_default_allocator() -> *mut sys::OrtAllocator {
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        unsafe {
            // this default non-arena allocator doesn't need to be deallocated
            call_ort(|ort| ort.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr))
        }
        .unwrap();
        allocator_ptr
    }
}
