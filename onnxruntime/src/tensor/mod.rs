//! Module containing tensor types.
//!
//! Two main types of tensors are available.
//!
//! The first one, [`Tensor`](struct.Tensor.html),
//! is an _owned_ tensor that is backed by [`ndarray`](https://crates.io/crates/ndarray).
//! This kind of tensor is used to pass input data for the inference.
//!
//! The second one, [`OrtOwnedTensor`](struct.OrtOwnedTensor.html), is used
//! internally to pass to the ONNX Runtime inference execution to place
//! its output values. It is built using a [`OrtOwnedTensorExtractor`](struct.OrtOwnedTensorExtractor.html)
//! following the builder pattern.
//!
//! Once "extracted" from the runtime environment, this tensor will contain an
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
//! containing _a view_ of the data. When going out of scope, this tensor will free the required
//! memory on the C side.
//!
//! **NOTE**: Tensors are not meant to be built directly. When performing inference,
//! the [`Session::run()`](../session/struct.Session.html#method.run) method takes
//! an `ndarray::Array` as input (taking ownership of it) and will convert it internally
//! to a [`Tensor`](struct.Tensor.html). After inference, a [`OrtOwnedTensor`](struct.OrtOwnedTensor.html)
//! will be returned by the method which can be derefed into its internal
//! [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).

pub mod ndarray_tensor;
pub mod ort_owned_tensor;
pub mod ort_tensor_dyn;

use std::fmt::Debug;

use ndarray::{ArrayBase, Data};

use crate::{error::Result, session::Session, TypeToTensorElementDataType};

pub use self::{
    ort_owned_tensor::OrtOwnedTensor,
    ort_tensor_dyn::{OrtTensorDyn, OrtTensorsDyn},
};

/// A list of different kinds of tensor references.
pub trait IntoOrtTensorsDyn<'t> {
    /// Convert the list of tensors into `OrtTensorsDyn`.
    fn into_ort_tensors_dyn<'m>(self, session: &'m Session) -> Result<OrtTensorsDyn<'t>>
    where
        'm: 't // 'm outlives 't
    ;
}

impl<'t> IntoOrtTensorsDyn<'t> for OrtTensorsDyn<'t> {
    fn into_ort_tensors_dyn<'m>(self, _: &'m Session) -> Result<OrtTensorsDyn<'t>>
    where
        'm: 't, // 'm outlives 't
    {
        Ok(self)
    }
}

impl<'t, T, Item> IntoOrtTensorsDyn<'t> for T
where
    T: IntoIterator<Item = Item>,
    Item: AsOrtTensorDyn<'t>,
{
    fn into_ort_tensors_dyn<'m>(self, session: &'m Session) -> Result<OrtTensorsDyn<'t>>
    where
        'm: 't, // 'm outlives 't
    {
        Ok(OrtTensorsDyn {
            inner: self
                .into_iter()
                .map(|tensor| tensor.as_ort_tensor_dyn(session))
                .collect::<Result<_>>()?,
        })
    }
}

/// A tensor reference.
pub trait AsOrtTensorDyn<'t> {
    /// Convert the tensors into `OrtTensorDyn`.
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> Result<OrtTensorDyn<'t>>
    where
        'm: 't // 'm outlives 't
    ;
}

impl<'t, T, D> AsOrtTensorDyn<'t> for ArrayBase<T, D>
where
    T: Data,
    T::Elem: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> Result<OrtTensorDyn<'t>>
    where
        'm: 't, // 'm outlives 't
    {
        OrtTensorDyn::from_array(&session.memory_info, session.allocator_ptr, self)
    }
}

impl<'t, T> AsOrtTensorDyn<'t> for &T
where
    T: AsOrtTensorDyn<'t>,
{
    fn as_ort_tensor_dyn<'m>(&self, session: &'m Session) -> Result<OrtTensorDyn<'t>>
    where
        'm: 't, // 'm outlives 't
    {
        (**self).as_ort_tensor_dyn(session)
    }
}
