use coker_bytecode::{RowOp, ScalarOp};

use crate::UNUSED_OPERAND;

pub(crate) fn evaluate_generic_value(row_operation: &RowOp, input_slice: &[f32]) -> f32 {
    let first = operand_value(row_operation.first, input_slice);
    let second = operand_value(row_operation.second, input_slice);
    let third = operand_value(row_operation.third, input_slice);

    match row_operation.op {
        ScalarOp::Identity => required_operand(first),
        ScalarOp::Sin => required_operand(first).sin(),
        ScalarOp::Cos => required_operand(first).cos(),
        ScalarOp::Tan => required_operand(first).tan(),
        ScalarOp::Exp => required_operand(first).exp(),
        ScalarOp::Sqrt => required_operand(first).sqrt(),
        ScalarOp::Log => required_operand(first).ln(),
        ScalarOp::Neg => -required_operand(first),
        ScalarOp::Abs => required_operand(first).abs(),
        ScalarOp::Add => required_operand(first) + required_operand(second),
        ScalarOp::Sub => required_operand(first) - required_operand(second),
        ScalarOp::Mul => required_operand(first) * required_operand(second),
        ScalarOp::Div => divide(required_operand(first), required_operand(second)),
        ScalarOp::Pow | ScalarOp::IntPow => required_operand(first).powf(required_operand(second)),
        ScalarOp::Atan2 => required_operand(first).atan2(required_operand(second)),
        ScalarOp::Equal => (required_operand(first) == required_operand(second)) as u8 as f32,
        ScalarOp::LessThan => (required_operand(first) < required_operand(second)) as u8 as f32,
        ScalarOp::LessEqual => (required_operand(first) <= required_operand(second)) as u8 as f32,
        ScalarOp::Case => {
            if required_operand(first) != 0.0 {
                required_operand(second)
            } else {
                required_operand(third)
            }
        }
    }
}

pub(crate) fn evaluate_generic_push_forward(
    row_operation: &RowOp,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
) -> (f32, f32) {
    let first = operand_value(row_operation.first, input_slice);
    let second = operand_value(row_operation.second, input_slice);
    let third = operand_value(row_operation.third, input_slice);
    let first_tangent = operand_tangent(row_operation.first, tangent_input_slice);
    let second_tangent = operand_tangent(row_operation.second, tangent_input_slice);
    let third_tangent = operand_tangent(row_operation.third, tangent_input_slice);

    match row_operation.op {
        ScalarOp::Identity => (required_operand(first), required_operand(first_tangent)),
        ScalarOp::Sin => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            (value.sin(), value.cos() * tangent)
        }
        ScalarOp::Cos => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            (value.cos(), -value.sin() * tangent)
        }
        ScalarOp::Tan => {
            let value = required_operand(first);
            let tangent = required_operand(first_tangent);
            (value.tan(), tangent / (value.cos() * value.cos()))
        }
        ScalarOp::Exp => {
            let value = required_operand(first).exp();
            (value, value * required_operand(first_tangent))
        }
        ScalarOp::Sqrt => {
            let value = required_operand(first).sqrt();
            (value, required_operand(first_tangent) / (2.0 * value))
        }
        ScalarOp::Log => {
            let value = required_operand(first);
            (value.ln(), required_operand(first_tangent) / value)
        }
        ScalarOp::Neg => (-required_operand(first), -required_operand(first_tangent)),
        ScalarOp::Abs => {
            let value = required_operand(first);
            (
                value.abs(),
                value.signum() * required_operand(first_tangent),
            )
        }
        ScalarOp::Add => (
            required_operand(first) + required_operand(second),
            required_operand(first_tangent) + required_operand(second_tangent),
        ),
        ScalarOp::Sub => (
            required_operand(first) - required_operand(second),
            required_operand(first_tangent) - required_operand(second_tangent),
        ),
        ScalarOp::Mul => {
            let first_value = required_operand(first);
            let second_value = required_operand(second);
            (
                first_value * second_value,
                second_value * required_operand(first_tangent)
                    + first_value * required_operand(second_tangent),
            )
        }
        ScalarOp::Div => {
            let numerator = required_operand(first);
            let denominator = required_operand(second);
            if denominator == 0.0 {
                return (f32::NAN, f32::NAN);
            }
            (
                divide(numerator, denominator),
                divide(
                    required_operand(first_tangent) * denominator
                        - numerator * required_operand(second_tangent),
                    denominator * denominator,
                ),
            )
        }
        ScalarOp::Pow | ScalarOp::IntPow => {
            let base = required_operand(first);
            let exponent = required_operand(second);
            let value = base.powf(exponent);
            if base == 0.0 {
                return (value, 0.0);
            }
            (
                value,
                value
                    * (required_operand(second_tangent) * base.ln()
                        + exponent * required_operand(first_tangent) / base),
            )
        }
        ScalarOp::Atan2 => {
            let first_value = required_operand(first);
            let second_value = required_operand(second);
            let denominator = first_value * first_value + second_value * second_value;
            (
                first_value.atan2(second_value),
                (second_value * required_operand(first_tangent)
                    - first_value * required_operand(second_tangent))
                    / denominator,
            )
        }
        ScalarOp::Equal => (
            (required_operand(first) == required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::LessThan => (
            (required_operand(first) < required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::LessEqual => (
            (required_operand(first) <= required_operand(second)) as u8 as f32,
            0.0,
        ),
        ScalarOp::Case => {
            if required_operand(first) != 0.0 {
                (required_operand(second), required_operand(second_tangent))
            } else {
                (required_operand(third), required_operand(third_tangent))
            }
        }
    }
}

fn operand_value(operand_index: u16, input_slice: &[f32]) -> Option<f32> {
    if operand_index == UNUSED_OPERAND {
        return None;
    }
    Some(input_slice[operand_index as usize])
}

fn operand_tangent(operand_index: u16, tangent_input_slice: &[f32]) -> Option<f32> {
    if operand_index == UNUSED_OPERAND {
        return None;
    }
    Some(tangent_input_slice[operand_index as usize])
}

fn required_operand(operand_value: Option<f32>) -> f32 {
    operand_value.expect("validated generic operation missing required operand")
}

pub(crate) fn homogeneous_value(input_slice: &[f32], index: u16) -> f32 {
    if index == 0 {
        return 1.0;
    }
    let offset = index as usize - 1;
    input_slice[offset]
}

pub(crate) fn homogeneous_tangent(tangent_input_slice: &[f32], index: u16) -> f32 {
    if index == 0 {
        return 0.0;
    }
    let offset = index as usize - 1;
    tangent_input_slice[offset]
}

fn divide(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        f32::NAN
    } else {
        numerator / denominator
    }
}
