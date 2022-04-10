//! Module containing tensor with memory owned by the ONNX Runtime

use std::{fmt::Debug, ops::Deref};

use ndarray::ArrayView;
use tracing::debug;

use onnxruntime_sys as sys;

use crate::{error::status_to_result, g_ort, OrtError, Result, TypeToTensorElementDataType};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
pub struct OrtOwnedTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    array_view: ArrayView<'t, T, D>,
}

unsafe impl<'t, T, D> Send for OrtOwnedTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
}

impl<'t, T, D> Deref for OrtOwnedTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array_view
    }
}

#[derive(Debug)]
pub(crate) struct OrtOwnedTensorExtractor<D>
where
    D: ndarray::Dimension,
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    shape: D,
}

impl<D> OrtOwnedTensorExtractor<D>
where
    D: ndarray::Dimension,
{
    pub(crate) fn new(shape: D) -> OrtOwnedTensorExtractor<D> {
        OrtOwnedTensorExtractor {
            tensor_ptr: std::ptr::null_mut(),
            shape,
        }
    }

    pub(crate) fn extract<'t, T>(self) -> Result<OrtOwnedTensor<'t, T, D>>
    where
        T: TypeToTensorElementDataType + Debug + Clone,
    {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.

        assert_ne!(self.tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        (is_tensor == 1)
            .then(|| ())
            .ok_or(OrtError::IsTensorCheck)?;

        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr as *mut *mut std::ffi::c_void;
        let status = unsafe {
            g_ort().GetTensorMutableData.unwrap()(self.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let array_view = unsafe { ArrayView::from_shape_ptr(self.shape, output_array_ptr) };

        Ok(OrtOwnedTensor {
            tensor_ptr: self.tensor_ptr,
            array_view,
        })
    }
}

impl<'t, T, D> Drop for OrtOwnedTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping OrtOwnedTensor.");
        unsafe { g_ort().ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = std::ptr::null_mut();
    }
}
