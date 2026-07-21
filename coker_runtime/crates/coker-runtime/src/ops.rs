use coker_bytecode::{RowOp, ScalarOp};

use crate::{RuntimeError, UNUSED_OPERAND};

pub(crate) fn evaluate_generic_value(
    row_operation: &RowOp,
    input_slice: &[f32],
) -> Result<f32, RuntimeError> {
    let first = operand_value(row_operation.first, input_slice)?;
    let second = operand_value(row_operation.second, input_slice)?;
    let third = operand_value(row_operation.third, input_slice)?;

    match row_operation.op {
        ScalarOp::Identity => Ok(required_operand(first, "identity")?),
        ScalarOp::Sin => Ok(required_operand(first, "sin")?.sin()),
        ScalarOp::Cos => Ok(required_operand(first, "cos")?.cos()),
        ScalarOp::Tan => Ok(required_operand(first, "tan")?.tan()),
        ScalarOp::Exp => Ok(required_operand(first, "exp")?.exp()),
        ScalarOp::Sqrt => Ok(required_operand(first, "sqrt")?.sqrt()),
        ScalarOp::Log => Ok(required_operand(first, "log")?.ln()),
        ScalarOp::Neg => Ok(-required_operand(first, "neg")?),
        ScalarOp::Abs => Ok(required_operand(first, "abs")?.abs()),
        ScalarOp::Add => {
            Ok(required_operand(first, "add")? + required_operand(second, "add")?)
        }
        ScalarOp::Sub => {
            Ok(required_operand(first, "sub")? - required_operand(second, "sub")?)
        }
        ScalarOp::Mul => {
            Ok(required_operand(first, "mul")? * required_operand(second, "mul")?)
        }
        ScalarOp::Div => Ok(divide(
            required_operand(first, "div")?,
            required_operand(second, "div")?,
        )),
        ScalarOp::Pow | ScalarOp::IntPow => {
            Ok(required_operand(first, "pow")?.powf(required_operand(second, "pow")?))
        }
        ScalarOp::Atan2 => Ok(
            required_operand(first, "atan2")?
                .atan2(required_operand(second, "atan2")?),
        ),
        ScalarOp::Equal => Ok(
            (required_operand(first, "equal")? == required_operand(second, "equal")?)
                as u8 as f32,
        ),
        ScalarOp::LessThan => Ok(
            (required_operand(first, "less_than")?
                < required_operand(second, "less_than")?) as u8 as f32,
        ),
        ScalarOp::LessEqual => Ok(
            (required_operand(first, "less_equal")?
                <= required_operand(second, "less_equal")?) as u8 as f32,
        ),
        ScalarOp::Case => {
            if required_operand(first, "case")? != 0.0 {
                Ok(required_operand(second, "case")?)
            } else {
                Ok(required_operand(third, "case")?)
            }
        }
    }
}

pub(crate) fn evaluate_generic_push_forward(
    row_operation: &RowOp,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
) -> Result<(f32, f32), RuntimeError> {
    let first = operand_value(row_operation.first, input_slice)?;
    let second = operand_value(row_operation.second, input_slice)?;
    let third = operand_value(row_operation.third, input_slice)?;
    let first_tangent = operand_tangent(row_operation.first, tangent_input_slice)?;
    let second_tangent = operand_tangent(row_operation.second, tangent_input_slice)?;
    let third_tangent = operand_tangent(row_operation.third, tangent_input_slice)?;

    match row_operation.op {
        ScalarOp::Identity => Ok((
            required_operand(first, "identity")?,
            required_operand(first_tangent, "identity")?,
        )),
        ScalarOp::Sin => {
            let value = required_operand(first, "sin")?;
            let tangent = required_operand(first_tangent, "sin")?;
            Ok((value.sin(), value.cos() * tangent))
        }
        ScalarOp::Cos => {
            let value = required_operand(first, "cos")?;
            let tangent = required_operand(first_tangent, "cos")?;
            Ok((value.cos(), -value.sin() * tangent))
        }
        ScalarOp::Tan => {
            let value = required_operand(first, "tan")?;
            let tangent = required_operand(first_tangent, "tan")?;
            Ok((value.tan(), tangent / (value.cos() * value.cos())))
        }
        ScalarOp::Exp => {
            let value = required_operand(first, "exp")?.exp();
            Ok((value, value * required_operand(first_tangent, "exp")?))
        }
        ScalarOp::Sqrt => {
            let value = required_operand(first, "sqrt")?.sqrt();
            Ok((value, required_operand(first_tangent, "sqrt")? / (2.0 * value)))
        }
        ScalarOp::Log => {
            let value = required_operand(first, "log")?;
            Ok((value.ln(), required_operand(first_tangent, "log")? / value))
        }
        ScalarOp::Neg => Ok((
            -required_operand(first, "neg")?,
            -required_operand(first_tangent, "neg")?,
        )),
        ScalarOp::Abs => {
            let value = required_operand(first, "abs")?;
            Ok((value.abs(), value.signum() * required_operand(first_tangent, "abs")?))
        }
        ScalarOp::Add => Ok((
            required_operand(first, "add")? + required_operand(second, "add")?,
            required_operand(first_tangent, "add")?
                + required_operand(second_tangent, "add")?,
        )),
        ScalarOp::Sub => Ok((
            required_operand(first, "sub")? - required_operand(second, "sub")?,
            required_operand(first_tangent, "sub")?
                - required_operand(second_tangent, "sub")?,
        )),
        ScalarOp::Mul => {
            let first_value = required_operand(first, "mul")?;
            let second_value = required_operand(second, "mul")?;
            Ok((
                first_value * second_value,
                second_value * required_operand(first_tangent, "mul")?
                    + first_value * required_operand(second_tangent, "mul")?,
            ))
        }
        ScalarOp::Div => {
            let numerator = required_operand(first, "div")?;
            let denominator = required_operand(second, "div")?;
            if denominator == 0.0 {
                return Ok((f32::NAN, f32::NAN));
            }
            Ok((
                divide(numerator, denominator),
                divide(
                    required_operand(first_tangent, "div")? * denominator
                        - numerator * required_operand(second_tangent, "div")?,
                    denominator * denominator,
                ),
            ))
        }
        ScalarOp::Pow | ScalarOp::IntPow => {
            let base = required_operand(first, "pow")?;
            let exponent = required_operand(second, "pow")?;
            let value = base.powf(exponent);
            if base == 0.0 {
                return Ok((value, 0.0));
            }
            Ok((
                value,
                value
                    * (required_operand(second_tangent, "pow")? * base.ln()
                        + exponent * required_operand(first_tangent, "pow")? / base),
            ))
        }
        ScalarOp::Atan2 => {
            let first_value = required_operand(first, "atan2")?;
            let second_value = required_operand(second, "atan2")?;
            let denominator = first_value * first_value + second_value * second_value;
            Ok((
                first_value.atan2(second_value),
                (second_value * required_operand(first_tangent, "atan2")?
                    - first_value * required_operand(second_tangent, "atan2")?)
                    / denominator,
            ))
        }
        ScalarOp::Equal => Ok((
            (required_operand(first, "equal")? == required_operand(second, "equal")?)
                as u8 as f32,
            0.0,
        )),
        ScalarOp::LessThan => Ok((
            (required_operand(first, "less_than")?
                < required_operand(second, "less_than")?) as u8 as f32,
            0.0,
        )),
        ScalarOp::LessEqual => Ok((
            (required_operand(first, "less_equal")?
                <= required_operand(second, "less_equal")?) as u8 as f32,
            0.0,
        )),
        ScalarOp::Case => {
            if required_operand(first, "case")? != 0.0 {
                Ok((
                    required_operand(second, "case")?,
                    required_operand(second_tangent, "case")?,
                ))
            } else {
                Ok((
                    required_operand(third, "case")?,
                    required_operand(third_tangent, "case")?,
                ))
            }
        }
    }
}

fn operand_value(
    operand_index: u16,
    input_slice: &[f32],
) -> Result<Option<f32>, RuntimeError> {
    if operand_index == UNUSED_OPERAND {
        return Ok(None);
    }
    Ok(Some(input_slice[operand_index as usize]))
}

fn operand_tangent(
    operand_index: u16,
    tangent_input_slice: &[f32],
) -> Result<Option<f32>, RuntimeError> {
    if operand_index == UNUSED_OPERAND {
        return Ok(None);
    }
    Ok(Some(tangent_input_slice[operand_index as usize]))
}

fn required_operand(
    operand_value: Option<f32>,
    context: &'static str,
) -> Result<f32, RuntimeError> {
    operand_value.ok_or_else(|| {
        RuntimeError::Validation(format!(
            "missing operand for generic operation {context}"
        ))
    })
}

pub(crate) fn homogeneous_value(input_slice: &[f32], index: u16) -> Result<f32, RuntimeError> {
    if index == 0 {
        return Ok(1.0);
    }
    let offset = index as usize - 1;
    Ok(input_slice[offset])
}

pub(crate) fn homogeneous_tangent(
    tangent_input_slice: &[f32],
    index: u16,
) -> Result<f32, RuntimeError> {
    if index == 0 {
        return Ok(0.0);
    }
    let offset = index as usize - 1;
    Ok(tangent_input_slice[offset])
}

fn divide(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        f32::NAN
    } else {
        numerator / denominator
    }
}
