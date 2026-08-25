import pytest

import contract


def _job(**overrides):
    base = {
        "op": "image_preprocess",
        "input_key": "in/test.png",
        "output_key": "out/test.jpg",
    }
    base.update(overrides)
    return base


def test_valid_job_normalizes_defaults():
    job = contract.validate_job(_job())
    assert job["op"] == "image_preprocess"
    assert job["params"] == {
        "target_max_dim": contract.DEFAULT_TARGET_DIM,
        "format": "jpeg",
        "quality": contract.DEFAULT_QUALITY,
    }


def test_non_dict_input_refused():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(None)
    assert exc.value.code == "invalid-input"


def test_unknown_op_refused():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(op="train_model"))
    assert exc.value.code == "op-not-allowed"


def test_no_model_field_exists_in_contract():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(model="anything"))
    assert exc.value.code == "invalid-input"
    assert "model" in exc.value.message


def test_no_gpu_field_exists_in_contract():
    with pytest.raises(contract.ContractError) as exc:
        contract.validate_job(_job(gpu="H100"))
    assert exc.value.code == "invalid-input"


def test_missing_keys_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job({"op": "image_preprocess"})


def test_key_with_traversal_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(input_key="a/../../etc/passwd"))


def test_key_with_leading_slash_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(output_key="/absolute/path"))


def test_key_too_long_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(input_key="a" * 513))


def test_unknown_param_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"steps": 50}))


def test_target_dim_out_of_bounds_refused():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": contract.MAX_TARGET_DIM + 1}))
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": contract.MIN_TARGET_DIM - 1}))


def test_bool_is_not_an_int_for_bounds():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"target_max_dim": True}))


def test_format_whitelist():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"format": "tiff"}))
    job = contract.validate_job(_job(params={"format": "webp"}))
    assert job["params"]["format"] == "webp"


def test_quality_bounds():
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"quality": 0}))
    with pytest.raises(contract.ContractError):
        contract.validate_job(_job(params={"quality": 101}))


def test_runtime_ceiling_is_900():
    assert contract.RUNTIME_CEILING_SECONDS == 900


def test_input_byte_bound_is_positive_and_finite():
    assert 0 < contract.MAX_INPUT_BYTES <= 64 * 1024 * 1024


def test_filter_output_drops_everything_not_whitelisted():
    filtered = contract.filter_output(
        {"ok": True, "device": "cuda", "aws_secret": "LEAK", "env": {"x": 1}}
    )
    assert filtered == {"ok": True, "device": "cuda"}


def test_output_whitelist_is_explicit_and_closed():
    assert "aws_secret" not in contract.OUTPUT_WHITELIST
    assert {"ok", "code", "error", "device", "gpu_name"} <= contract.OUTPUT_WHITELIST
