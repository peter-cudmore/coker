use super::*;

pub(super) fn program_info_from_program(
    program: &ArchivedProgram,
) -> ProgramInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
    ProgramInfo {
        workspace_size: us32(program.workspace_size),
        required_workspace_size: us32(program.required_workspace_size),
        input_specs: program.input_specs(),
        output_specs: program.output_specs(),
    }
}

pub(super) fn entry_program(
    module: &ArchivedBytecodeModule,
) -> Result<&ArchivedProgram, RuntimeError> {
    find_function(module, 0).ok_or(RuntimeError::Validation("missing entry function_id 0"))
}

pub(super) fn find_function(
    module: &ArchivedBytecodeModule,
    function_id: u16,
) -> Option<&ArchivedProgram> {
    module.program(function_id)
}

pub(super) fn find_function_unchecked(
    module: &ArchivedBytecodeModule,
    function_id: u16,
) -> &ArchivedProgram {
    find_function(module, function_id).expect("validated module missing referenced function")
}

pub(super) fn row_op_from_archived(row_op: &ArchivedRowOp) -> RowOp {
    RowOp {
        first: u16n(row_op.first),
        second: u16n(row_op.second),
        third: u16n(row_op.third),
        op: scalar_op_from_archived(&row_op.op),
    }
}

pub(super) fn scalar_op_from_archived(scalar_op: &ArchivedScalarOp) -> ScalarOp {
    match scalar_op {
        ArchivedScalarOp::Identity => ScalarOp::Identity,
        ArchivedScalarOp::Sin => ScalarOp::Sin,
        ArchivedScalarOp::Cos => ScalarOp::Cos,
        ArchivedScalarOp::Tan => ScalarOp::Tan,
        ArchivedScalarOp::Exp => ScalarOp::Exp,
        ArchivedScalarOp::Sqrt => ScalarOp::Sqrt,
        ArchivedScalarOp::Log => ScalarOp::Log,
        ArchivedScalarOp::Neg => ScalarOp::Neg,
        ArchivedScalarOp::Abs => ScalarOp::Abs,
        ArchivedScalarOp::Add => ScalarOp::Add,
        ArchivedScalarOp::Sub => ScalarOp::Sub,
        ArchivedScalarOp::Mul => ScalarOp::Mul,
        ArchivedScalarOp::Div => ScalarOp::Div,
        ArchivedScalarOp::Pow => ScalarOp::Pow,
        ArchivedScalarOp::IntPow => ScalarOp::IntPow,
        ArchivedScalarOp::Atan2 => ScalarOp::Atan2,
        ArchivedScalarOp::Equal => ScalarOp::Equal,
        ArchivedScalarOp::LessThan => ScalarOp::LessThan,
        ArchivedScalarOp::LessEqual => ScalarOp::LessEqual,
        ArchivedScalarOp::Case => ScalarOp::Case,
    }
}

pub(super) fn u16n(value: impl Into<u16>) -> u16 {
    value.into()
}

pub(super) fn u32n(value: impl Into<u32>) -> u32 {
    value.into()
}

pub(super) fn us16(value: impl Into<u16>) -> usize {
    u16n(value) as usize
}

pub(super) fn us32(value: impl Into<u32>) -> usize {
    u32n(value) as usize
}

pub(super) fn f32n(value: impl Into<f32>) -> f32 {
    value.into()
}
