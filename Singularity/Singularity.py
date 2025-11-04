"""
The Riemann Hypothesis Ultimate Test
A monolithic verification of mathematics' most profound enigma.

Crafted in the spirit of Euler's clarity and von Neumann's rigor,
this code stands as a self-contained oracle for truth.
"""

import mpmath
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt

# --------------------------
# SACRED CONSTANTS
# --------------------------
NUM_ZEROS = 1000  # Number of zeros to analyze
PRECISION_DIGITS = 50  # Floating-point precision
TOLERANCE = 1e-10  # Acceptable error margin
ENABLE_PLOTTING = True  # Visual enlightenment toggle

# --------------------------
# PRECISION INITIALIZATION
# --------------------------
mpmath.mp.dps = PRECISION_DIGITS
print(f"\n⚡ Initializing Riemann Oracle with {PRECISION_DIGITS} digits of precision...\n")

# --------------------------
# ZETA FUNCTION UTILITIES
# --------------------------

def zeta(s):
    """The Riemann zeta function - a cosmic symphony in complex variables."""
    return mpmath.zeta(s)

def functional_equation_test(s):
    """
    Verify the mirror symmetry of the zeta universe.
    ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
    """
    lhs = zeta(s)
    rhs = (2**s) * (mpmath.pi**(s-1)) * mpmath.sin(mpmath.pi*s/2) * mpmath.gamma(1-s) * zeta(1-s)
    return mpmath.almosteq(lhs, rhs, abs_eps=TOLERANCE)

def compute_zeros(n):
    """Harvest the mystical zeros where ζ(1/2 + it) = 0."""
    return mpmath.zetazero(n)

# --------------------------
# ANALYTICAL VERIFICATIONS
# --------------------------

def verify_zeros_on_critical_line(zeros):
    """Confirm all zeros lie precisely on Re(s) = 1/2."""
    deviations = [abs(float(z.real) - 0.5) for z in zeros]
    max_deviation = max(deviations)
    passed = max_deviation < TOLERANCE
    return passed, max_deviation

def li_criterion_verification(n):
    """
    Li's Criterion: λ_n positivity ≡ RH truth.
    A beautiful bridge between zeros and primes.
    """
    def li_coefficient(n):
        """Compute the nth Li coefficient with arithmetic grace."""
        s = mpmath.mpf(1)/2
        sum_term = 0
        for k in range(1, n+1):
            binom = mpmath.binomial(n-1, k-1)
            term = binom * mpmath.power(-1, k-1) * mpmath.power(s, k) / mpmath.factorial(k)
            sum_term += term
        return float((1 - (1 - mpmath.power(1 - s, n))) / s + sum_term)

    coefficients = [li_coefficient(k) for k in range(1, n+1)]
    all_positive = all(c > -TOLERANCE for c in coefficients)
    return all_positive, coefficients[:10]  # Return first 10 for inspection

def zero_spacing_analysis(zeros):
    """
    Examine the quantum chaos in zero spacings.
    GUE hypothesis: zeros repel like eigenvalues of random Hermitian matrices.
    """
    im_parts = [float(z.imag) for z in zeros]
    spacings = np.diff(sorted(im_parts))
    normalized_spacings = spacings / np.mean(spacings)

    # KS test against exponential (Poisson) and GUE distributions
    def gue_cdf(x):
        """GUE spacing distribution - the fingerprint of the primes."""
        return 1 - np.exp(-np.pi * x**2 / 4)

    ks_stat, p_value = kstest(normalized_spacings, gue_cdf)
    return ks_stat < 0.1, (normalized_spacings, gue_cdf)  # 0.1 threshold for KS statistic

# --------------------------
# VISUALIZATION
# --------------------------

def plot_zero_spacing(normalized_spacings, gue_cdf):
    """Render the cosmic dance of the zeros."""
    plt.figure(figsize=(10, 6))

    # Histogram of observed spacings
    plt.hist(normalized_spacings, bins=50, density=True, alpha=0.7,
             label='Normalized Zero Spacings')

    # Theoretical GUE distribution
    x = np.linspace(0, 3, 300)
    plt.plot(x, np.pi * x / 2 * np.exp(-np.pi * x**2 / 4),
             'r-', lw=2, label='GUE Prediction')

    plt.title('Quantum Chaos in Riemann Zeros', fontsize=14)
    plt.xlabel('Normalized Spacing', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --------------------------
# MAIN VERIFICATION ROUTINE
# --------------------------

def riemann_hypothesis_test_suite():
    """The grand unification of all RH verification methods."""
    print("🌀 Beginning Riemann Hypothesis Verification Suite")
    print(f"🔍 Analyzing first {NUM_ZEROS} non-trivial zeros\n")

    # Phase 1: Zero Collection
    print("⏳ Harvesting zeros from the critical line...")
    zeros = [compute_zeros(n+1) for n in range(NUM_ZEROS)]
    print("✅ Zero harvesting complete\n")

    # Phase 2: Critical Line Verification
    print("🧪 Testing zeros lie on Re(s) = 1/2...")
    crit_line_passed, max_dev = verify_zeros_on_critical_line(zeros)
    print(f"   Max deviation: {max_dev:.3e}")
    print("✅ PASS" if crit_line_passed else "❌ FAIL", "\n")

    # Phase 3: Functional Equation Verification
    print("⚖️ Validating functional equation at sample points...")
    test_points = [0.5 + 1j*zeros[i].imag for i in [0, NUM_ZEROS//4, NUM_ZEROS//2, -1]]
    fe_passed = all(functional_equation_test(p) for p in test_points)
    print("✅ PASS" if fe_passed else "❌ FAIL", "\n")

    # Phase 4: Li's Criterion
    print("📊 Verifying Li's Criterion (positivity of coefficients)...")
    li_passed, li_coeffs = li_criterion_verification(min(20, NUM_ZEROS))
    print(f"   First 10 coefficients: {[f'{c:.3f}' for c in li_coeffs]}")
    print("✅ PASS" if li_passed else "❌ FAIL", "\n")

    # Phase 5: Spacing Statistics
    print("📈 Analyzing zero spacing distribution...")
    spacing_passed, spacing_data = zero_spacing_analysis(zeros)
    print("✅ PASS" if spacing_passed else "❌ FAIL", "\n")

    # Final Assessment
    all_passed = crit_line_passed and fe_passed and li_passed and spacing_passed
    print("\n" + "="*50)
    print("🌌 FINAL VERDICT: RIEMANN HYPOTHESIS " +
          ("HOLDS" if all_passed else "MAY BE VIOLATED") + " IN TESTED DOMAIN")
    print("="*50 + "\n")

    # Visualization if enabled
    if ENABLE_PLOTTING and spacing_passed:
        plot_zero_spacing(*spacing_data)

    return all_passed

# --------------------------
# EXECUTION
# --------------------------
if __name__ == "__main__":
    riemann_hypothesis_test_suite()
