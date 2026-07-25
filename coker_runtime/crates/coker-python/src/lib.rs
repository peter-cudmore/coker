#![cfg(feature = "std")]

use core::{mem::MaybeUninit, slice};

use coker_bytecode::archived_module;
use coker_compiler::{compile_exported_json, compile_exported_qp_json, CompileError};
use coker_runtime::{
    program_info, validate_module, MappedModule, MappedQpProgram, MappedQpPushForwardWorkspace,
    MappedQpWorkspace, Module, ModuleBuilder, ProgramInfo, QpSolveStatus, QpWorkspaceRequirements,
    SpecInfo,
};
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass(name = "RuntimeProgram")]
struct PyRuntimeProgram {
    module: Module,
    workspace_size: usize,
    output_lengths: Vec<usize>,
}

#[pyclass(name = "RuntimeQpProgram", unsendable)]
struct PyRuntimeQpProgram {
    module_bytes: Vec<u8>,
    function_id: u16,
    input_lengths: Vec<usize>,
    output_length: usize,
    workspace_requirements: QpWorkspaceRequirements,
}

#[pymethods]
impl PyRuntimeQpProgram {
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new(py);
        info.set_item("function_id", self.function_id)?;
        info.set_item("input_specs", &self.input_lengths)?;
        info.set_item("output_spec", self.output_length)?;
        Ok(info)
    }

    fn workspace_requirements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let requirements = PyDict::new(py);
        requirements.set_item(
            "evaluator_workspace_size",
            self.workspace_requirements.evaluator_workspace_size,
        )?;
        requirements.set_item(
            "tangent_workspace_size",
            self.workspace_requirements.tangent_workspace_size,
        )?;
        requirements.set_item(
            "coefficient_output_size",
            self.workspace_requirements.coefficient_output_size,
        )?;
        requirements.set_item("arena_bytes", self.workspace_requirements.arena_bytes)?;
        requirements.set_item(
            "arena_alignment",
            self.workspace_requirements.arena_alignment,
        )?;
        Ok(requirements)
    }

    #[pyo3(signature = (inputs, arena, evaluator_workspace, coefficient_outputs, warm_start=None))]
    fn solve(
        &self,
        _py: Python<'_>,
        inputs: Vec<Vec<f32>>,
        arena: PyBuffer<u8>,
        evaluator_workspace: PyBuffer<f32>,
        coefficient_outputs: PyBuffer<f32>,
        warm_start: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, bool, String)> {
        let arena = writable_c_buffer(&arena, "arena")?;
        let evaluator_workspace = writable_c_buffer(&evaluator_workspace, "evaluator_workspace")?;
        let coefficient_outputs = writable_c_buffer(&coefficient_outputs, "coefficient_outputs")?;
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let mapped_qp = self.mapped_qp_program().map_err(runtime_error)?;
        let workspace = MappedQpWorkspace::new(evaluator_workspace, coefficient_outputs);
        let mut bound = mapped_qp
            .bind(arena_as_uninit(arena))
            .map_err(runtime_error)?;
        let mut outputs = vec![0.0f32; self.output_length];
        let diagnostics = bound
            .execute(
                &input_slices,
                warm_start.as_deref(),
                workspace,
                &mut outputs,
            )
            .map_err(runtime_error)?;
        let success = solve_success(diagnostics.status);
        let solution = outputs.into_iter().map(f64::from).collect();
        Ok((solution, success, format!("{:?}", diagnostics.status)))
    }

    #[allow(clippy::too_many_arguments)]
    fn push_forward(
        &self,
        _py: Python<'_>,
        inputs: Vec<Vec<f32>>,
        tangents: Vec<Vec<f32>>,
        arena: PyBuffer<u8>,
        primal_evaluator_workspace: PyBuffer<f32>,
        primal_coefficient_outputs: PyBuffer<f32>,
        tangent_evaluator_workspace: PyBuffer<f32>,
        tangent_coefficient_outputs: PyBuffer<f32>,
        solution_tangent_workspace: PyBuffer<f32>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let arena = writable_c_buffer(&arena, "arena")?;
        let primal_evaluator_workspace =
            writable_c_buffer(&primal_evaluator_workspace, "primal_evaluator_workspace")?;
        let primal_coefficient_outputs =
            writable_c_buffer(&primal_coefficient_outputs, "primal_coefficient_outputs")?;
        let tangent_evaluator_workspace =
            writable_c_buffer(&tangent_evaluator_workspace, "tangent_evaluator_workspace")?;
        let tangent_coefficient_outputs =
            writable_c_buffer(&tangent_coefficient_outputs, "tangent_coefficient_outputs")?;
        let solution_tangent_workspace =
            writable_c_buffer(&solution_tangent_workspace, "solution_tangent_workspace")?;
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let tangent_slices: Vec<&[f32]> = tangents.iter().map(|input| input.as_slice()).collect();
        let mapped_qp = self.mapped_qp_program().map_err(runtime_error)?;
        let workspace = MappedQpPushForwardWorkspace::new(
            primal_evaluator_workspace,
            primal_coefficient_outputs,
            tangent_evaluator_workspace,
            tangent_coefficient_outputs,
            solution_tangent_workspace,
        );
        let mut bound = mapped_qp
            .bind(arena_as_uninit(arena))
            .map_err(runtime_error)?;
        let mut outputs = vec![0.0f32; self.output_length];
        let mut tangent_outputs = vec![0.0f32; self.output_length];
        bound
            .push_forward(
                &input_slices,
                &tangent_slices,
                workspace,
                &mut outputs,
                &mut tangent_outputs,
            )
            .map_err(runtime_error)?;
        Ok((
            outputs.into_iter().map(f64::from).collect(),
            tangent_outputs.into_iter().map(f64::from).collect(),
        ))
    }
}

impl PyRuntimeQpProgram {
    fn mapped_qp_program(&self) -> Result<MappedQpProgram<'_>, coker_runtime::RuntimeError> {
        MappedModule::new_from_bytes(&self.module_bytes)?.qp_program(self.function_id)
    }
}

#[pymethods]
impl PyRuntimeProgram {
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        program_info_dict(py, &self.module.info())
    }

    fn execute(&mut self, inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let mut workspace = vec![0.0; self.workspace_size];
        let mut outputs = vec![0.0; self.output_lengths.iter().sum()];
        self.module
            .execute(&input_slices, &mut workspace, &mut outputs)
            .map_err(runtime_error)?;
        Ok(outputs)
    }

    fn push_forward(
        &mut self,
        inputs: Vec<Vec<f32>>,
        tangents: Vec<Vec<f32>>,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let tangent_slices: Vec<&[f32]> = tangents.iter().map(|input| input.as_slice()).collect();
        let mut workspace = vec![0.0; self.workspace_size];
        let mut tangent_workspace = vec![0.0; self.workspace_size];
        let output_length: usize = self.output_lengths.iter().sum();
        let mut outputs = vec![0.0; output_length];
        let mut tangent_outputs = vec![0.0; output_length];
        self.module
            .push_forward(
                &input_slices,
                &tangent_slices,
                &mut workspace,
                &mut tangent_workspace,
                &mut outputs,
                &mut tangent_outputs,
            )
            .map_err(runtime_error)?;
        Ok((outputs, tangent_outputs))
    }
}

#[pyfunction]
fn compile_exported_graph<'py>(
    py: Python<'py>,
    exported_graph_json: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let module_bytes =
        compile_exported_json(exported_graph_json).map_err(compile_error_to_python)?;
    Ok(PyBytes::new(py, &module_bytes))
}

#[pyfunction]
fn compile_exported_qp<'py>(
    py: Python<'py>,
    exported_qp_json: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = compile_exported_qp_json(exported_qp_json).map_err(compile_error_to_python)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn load_qp_program(program: &[u8]) -> PyResult<PyRuntimeQpProgram> {
    let (function_id, input_lengths, output_length, workspace_requirements) =
        load_qp_program_metadata(program).map_err(runtime_error)?;
    Ok(PyRuntimeQpProgram {
        module_bytes: program.to_vec(),
        function_id,
        input_lengths,
        output_length,
        workspace_requirements,
    })
}

#[pyfunction]
fn load_program(program: &[u8]) -> PyResult<PyRuntimeProgram> {
    let module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let (workspace_size, output_lengths) = {
        let info = module.info();
        (info.required_workspace_size, output_lengths(&info))
    };
    Ok(PyRuntimeProgram {
        module,
        workspace_size,
        output_lengths,
    })
}

#[pyfunction]
fn validate_compiled_program(program: &[u8]) -> PyResult<bool> {
    validate_module(program)
        .map(|_| true)
        .map_err(runtime_error)
}

#[pyfunction(name = "program_info")]
fn program_info_py<'py>(py: Python<'py>, program: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let module = validate_module(program).map_err(runtime_error)?;
    let info = program_info(&module);
    program_info_dict(py, &info)
}

#[pyfunction]
fn execute_program(program: &[u8], inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let info = module.info();
    let mut workspace = vec![0.0; info.required_workspace_size];
    let output_length: usize = info
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .sum();
    let mut outputs = vec![0.0; output_length];
    module
        .execute(&input_slices, &mut workspace, &mut outputs)
        .map_err(runtime_error)?;
    Ok(outputs)
}

#[pyfunction]
fn push_forward_program(
    program: &[u8],
    inputs: Vec<Vec<f32>>,
    tangents: Vec<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let tangent_slices: Vec<&[f32]> = tangents.iter().map(|input| input.as_slice()).collect();
    let info = module.info();
    let mut workspace = vec![0.0; info.required_workspace_size];
    let mut tangent_workspace = vec![0.0; info.required_workspace_size];
    let output_length: usize = info
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .sum();
    let mut outputs = vec![0.0; output_length];
    let mut tangent_outputs = vec![0.0; output_length];
    module
        .push_forward(
            &input_slices,
            &tangent_slices,
            &mut workspace,
            &mut tangent_workspace,
            &mut outputs,
            &mut tangent_outputs,
        )
        .map_err(runtime_error)?;
    Ok((outputs, tangent_outputs))
}

fn output_lengths<I: SpecInfo, O: SpecInfo>(info: &ProgramInfo<'_, I, O>) -> Vec<usize> {
    info.output_specs
        .iter()
        .map(|output_spec| output_spec.length())
        .collect()
}

fn load_qp_program_metadata(
    program: &[u8],
) -> Result<(u16, Vec<usize>, usize, QpWorkspaceRequirements), coker_runtime::RuntimeError> {
    let archived = archived_module(program)?;
    let mut qp_programs = archived.qp_programs();
    let (function_id, _sole_qp_program) = qp_programs.next().ok_or_else(|| {
        coker_runtime::RuntimeError::Validation(
            "module contains no QP programs; expected exactly one".to_string(),
        )
    })?;
    if qp_programs.next().is_some() {
        return Err(coker_runtime::RuntimeError::Validation(
            "module contains multiple QP programs; expected exactly one".to_string(),
        ));
    }
    let mapped = MappedModule::new_from_bytes(program)?;
    let mapped_qp = mapped.qp_program(function_id)?;
    let info = mapped_qp.info();
    Ok((
        mapped_qp.function_id(),
        info.input_specs.iter().map(|spec| spec.length()).collect(),
        info.output_spec.length(),
        mapped_qp.workspace_requirements(),
    ))
}

#[allow(clippy::mut_from_ref)]
fn writable_c_buffer<'py, T: Element>(
    buffer: &'py PyBuffer<T>,
    name: &str,
) -> PyResult<&'py mut [T]> {
    if buffer.readonly() {
        return Err(PyValueError::new_err(format!(
            "{name} buffer must be writable"
        )));
    }
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} buffer must be C-contiguous"
        )));
    }
    Ok(unsafe { slice::from_raw_parts_mut(buffer.buf_ptr() as *mut T, buffer.item_count()) })
}

fn arena_as_uninit(arena: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    unsafe { slice::from_raw_parts_mut(arena.as_mut_ptr().cast::<MaybeUninit<u8>>(), arena.len()) }
}

fn solve_success(status: QpSolveStatus) -> bool {
    matches!(
        status,
        QpSolveStatus::Solved | QpSolveStatus::SolvedInaccurate
    )
}

fn program_info_dict<'py, I: SpecInfo, O: SpecInfo>(
    py: Python<'py>,
    info: &ProgramInfo<'_, I, O>,
) -> PyResult<Bound<'py, PyDict>> {
    let info_dict = PyDict::new(py);
    info_dict.set_item("workspace_size", info.workspace_size)?;
    info_dict.set_item("required_workspace_size", info.required_workspace_size)?;
    let input_specs = info
        .input_specs
        .iter()
        .map(|input_spec| input_spec.length())
        .collect::<Vec<_>>();
    let output_specs = info
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length())
        .collect::<Vec<_>>();
    info_dict.set_item("input_specs", input_specs)?;
    info_dict.set_item("output_specs", output_specs)?;
    Ok(info_dict)
}

fn compile_error_to_python(error: CompileError) -> PyErr {
    match error {
        CompileError::NotImplemented(message) => PyNotImplementedError::new_err(message),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn runtime_error(error: impl core::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
#[pyo3(name = "_coker_runtime")]
fn coker_python(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRuntimeProgram>()?;
    module.add_class::<PyRuntimeQpProgram>()?;
    module.add_function(wrap_pyfunction!(compile_exported_qp, module)?)?;
    module.add_function(wrap_pyfunction!(load_qp_program, module)?)?;
    module.add_function(wrap_pyfunction!(compile_exported_graph, module)?)?;
    module.add_function(wrap_pyfunction!(load_program, module)?)?;
    module.add_function(wrap_pyfunction!(validate_compiled_program, module)?)?;
    module.add_function(wrap_pyfunction!(program_info_py, module)?)?;
    module.add_function(wrap_pyfunction!(execute_program, module)?)?;
    module.add_function(wrap_pyfunction!(push_forward_program, module)?)?;
    Ok(())
}
