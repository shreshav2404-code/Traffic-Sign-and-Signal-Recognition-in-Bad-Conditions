"""Check the fuzzy controller on known cases, and against scikit-fuzzy's own engine."""
import numpy as np
from skfuzzy import control as ctrl

from fuzzy_logic import controller
from fuzzy_logic.membership import BRIGHTNESS, CONTRAST, INPUT_SETS, NOISE, STRENGTH, STRENGTH_SETS, degree
from fuzzy_logic.rules import OUTPUTS, RULES, describe
from image_processing import enhancement, quality

rng = np.random.default_rng(0)
passed = 0


def check(name, condition, detail=""):
    global passed
    if not condition:
        raise SystemExit(f"FAILED: {name} {detail}")
    passed += 1
    print(f"ok  {name} {detail}")


# Membership functions
check("dark at brightness 0.05", degree("brightness", "dark", 0.05) == 1)
check("half dark, half normal at 0.15", abs(degree("brightness", "dark", 0.15) - 0.5) < 0.01 and abs(degree("brightness", "normal", 0.15) - 0.5) < 0.01)
for variable, universe in (("brightness", BRIGHTNESS), ("contrast", CONTRAST), ("noise", NOISE)):
    total = sum(INPUT_SETS[variable].values())
    check(f"{variable} sets cover every value", total.min() > 0, f"lowest total {total.min():.2f}")

# Known cases
cases = {
    "clean sign": ((0.42, 0.35, 3), {"denoise": 0, "stretch": 0, "darken": 0}),
    "dark sign": ((0.05, 0.05, 2), {"denoise": 0, "stretch": 1, "darken": 0}),
    "hazy sign": ((0.53, 0.05, 2), {"denoise": 0, "stretch": 1, "darken": 0}),
    "overexposed sign": ((0.92, 0.3, 3), {"denoise": 0, "stretch": 0, "darken": 1}),
    "bright, flat haze": ((0.83, 0.03, 1), {"denoise": 0, "stretch": 1, "darken": 0}),
    "noisy sign": ((0.42, 0.35, 30), {"denoise": 1, "stretch": 0, "darken": 0}),
}
for name, (inputs, expected) in cases.items():
    result = controller.decide(*inputs)
    for output, value in expected.items():
        got = float(result[output][0])
        check(f"{name}: {output} is {value}", abs(got - value) < 0.02, f"got {got:.3f}")

batch = rng.random((200, 3)) * np.array([1, 1, 60])
together = controller.decide(batch[:, 0], batch[:, 1], batch[:, 2])
for row in range(0, 200, 40):
    single = controller.decide(*batch[row])
    check(f"batch matches single image, row {row}", all(abs(together[o][row] - single[o][0]) < 1e-9 for o in OUTPUTS))
check("outputs stay between 0 and 1", all(0 <= together[o].min() and together[o].max() <= 1 for o in OUTPUTS))

# Compare with scikit-fuzzy's own Mamdani engine, using the same sets and rules.
antecedents = {
    "brightness": ctrl.Antecedent(BRIGHTNESS, "brightness"),
    "contrast": ctrl.Antecedent(CONTRAST, "contrast"),
    "noise": ctrl.Antecedent(NOISE, "noise"),
}
for name, antecedent in antecedents.items():
    for label, values in INPUT_SETS[name].items():
        antecedent[label] = values
consequents = {o: ctrl.Consequent(STRENGTH, o, defuzzify_method="centroid") for o in OUTPUTS}
for consequent in consequents.values():
    for label, values in STRENGTH_SETS.items():
        consequent[label] = values

library_rules = []
for rule in RULES:
    terms = [antecedents[variable][label] for variable, label in rule.conditions]
    condition = terms[0]
    for term in terms[1:]:
        condition = condition & term
    library_rules.append(ctrl.Rule(condition, consequents[rule.output][rule.strength]))

simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(library_rules))
largest_gap = 0.0
for row in range(100):
    b, c, n = batch[row]
    simulation.input["brightness"] = b
    simulation.input["contrast"] = c
    simulation.input["noise"] = n
    simulation.compute()
    ours = controller.decide(b, c, n)
    for output in OUTPUTS:
        largest_gap = max(largest_gap, abs(np.clip(simulation.output[output], 0, 1) - ours[output][0]))
check("matches scikit-fuzzy on 100 random inputs", largest_gap < 0.01, f"largest gap {largest_gap:.4f}")

# Enhancement
image = rng.integers(0, 256, (32, 32, 3), np.uint8)
check("zero strengths change nothing", np.array_equal(enhancement.enhance(image), image))
dark = (image // 6).astype(np.uint8)
check("stretch brightens a dark image", quality.brightness(enhancement.enhance(dark, stretch=1)) > quality.brightness(dark) * 2)
check("half stretch sits between", quality.brightness(dark) < quality.brightness(enhancement.enhance(dark, stretch=0.5)) < quality.brightness(enhancement.enhance(dark, stretch=1)))
bright = np.clip(image.astype(int) + 120, 0, 255).astype(np.uint8)
check("darken lowers brightness", quality.brightness(enhancement.enhance(bright, darken=1)) < quality.brightness(bright))
check("denoise smooths the image", quality.noise_sigma(enhancement.enhance(image, denoise=1)) < quality.noise_sigma(image))

result, measurements, settings = controller.enhance_image(dark)
check("enhance_image returns matching sizes", result.shape == dark.shape and result.dtype == np.uint8)
print("    measurements", {k: round(v, 3) for k, v in measurements.items()})
print("    settings", {k: round(v, 3) for k, v in settings.items()})

print("\nRule base")
for number, rule in enumerate(RULES, 1):
    print(f"    R{number}: {describe(rule)}")
print(f"\nAll {passed} checks passed.")
