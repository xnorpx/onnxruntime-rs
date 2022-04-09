//! Module containing tensor with memory owned by Rust

use std::{ffi, fmt::Debug, ops::Deref};

use ndarray::ArrayView;
use tracing::{debug, error};

use onnxruntime_sys as sys;

use crate::{
    error::{assert_not_null_pointer, call_ort, status_to_result},
    g_ort,
    memory::MemoryInfo,
    OrtError, Result, TensorElementDataType, TypeToTensorElementDataType,
};

/// Owned tensor, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
///
/// This tensor bounds the ONNX Runtime to `ndarray`; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
#[derive(Debug)]
pub struct OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    pub(crate) c_ptr: *mut sys::OrtValue,
    array: ArrayView<'t, T, D>,
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    pub(crate) fn from_array<'m>(
        memory_info: &'m MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        array: ArrayView<'t, T, D>,
    ) -> Result<OrtTensor<'t, T, D>>
    where
        'm: 't, // 'm outlives 't
    {
        Ok(OrtTensor {
            c_ptr: super::ort_tensor_dyn::OrtTensorDyn::from_array_get_ptr(
                memory_info,
                allocator_ptr,
                &array,
            )?,
            array,
        })
    }
}

impl<'t, T, D> Deref for OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<'t, T, D> Drop for OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
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
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array.view()).unwrap();
        let expected_shape: &[usize] = &[];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_1d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array.view()).unwrap();
        let expected_shape: &[usize] = &[6];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_2d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array.view()).unwrap();
        assert_eq!(tensor.shape(), &[2, 6]);
    }

    #[test]
    fn orttensor_from_array_3d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array.view()).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 6]);
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
            OrtTensor::from_array(&memory_info, ort_default_allocator(), array.view()).unwrap();
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn orttensor_from_array_3d_str() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [["1", "2", "3"], ["4", "5", "6"]],
            [["7", "8", "9"], ["10", "11", "12"]],
        ]);
        let tensor =
            OrtTensor::from_array(&memory_info, ort_default_allocator(), array.view()).unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 3]);
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
