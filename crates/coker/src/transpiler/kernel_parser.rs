use pyo3::prelude::*;
use crate::errors::CokerError;
use crate::project::Project;


pub fn compile_one(project: Project, target: &String) -> Result<(), CokerError> {
    
    Ok(())
}

pub fn compile_all(project: Project) -> Result<(), CokerError> {
    Ok(())
}

struct DenseArray{
    data: Vec<f32>,
    shape: Vec<usize>,
}

struct SparseArray{
    // Compressed Col Space
    data: Vec<f32>,
    row_index: Vec<usize>,
    col_start: Vec<usize>,
}


enum ValueType {
    Float(f64),
    Int(i64),
    DenseArray(DenseArray),
    SparseArray(SparseArray),
}




fn transpile_kernel<'py>(py:Python<'py>, kernel: Bound<'py, PyObject>){

    // flatten input space
    // by construction inputs: PI: L[X_1 * X_2 *...* X_n, R^k]
    //and PI^{-1}_i: L[R^k, X_i]
    
    // construct output projections:
    // PO: L[]
    
}