import mpmath
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Configure ultra-high precision environment
    mpmath.mp.dps = 40
    print(f"🚀 Critical Shell Scanner engaged (mpmath precision: {mpmath.mp.dps} digits)\n")

    # Scanning parameters
    t_min, t_max = 10.0, 1000.0  # Height range
    num_t_points = 100            # Number of heights to check
    c = 0.2                       # Zero-free region constant
    epsilon = 1e-12               # Zero detection threshold
    sigma_margin = 0.01           # How far inside boundary to scan

    # Generate logarithmic spaced t values
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), num_t_points)

    # Storage for potential violations
    violations = []
    min_zeta_mag = float('inf')
    min_zeta_loc = None

    print("🔍 Scanning zero-free boundary...")
    for t in t_values:
        # Calculate current zero-free boundary: σ > 1 - c/log(t)
        boundary = 1 - c / mpmath.log(t)

        # Sample points just inside the boundary (σ = boundary + margin)
        sigma_test = boundary + sigma_margin

        # Evaluate ζ(s) at test point
        s = mpmath.mpc(sigma_test, t)
        zeta_val = mpmath.zeta(s)
        zeta_mag = abs(zeta_val)

        # Record minimum magnitude encountered
        if zeta_mag < min_zeta_mag:
            min_zeta_mag = float(zeta_mag)
            min_zeta_loc = (float(sigma_test), float(t))

        # Flag potential violations
        if zeta_mag < epsilon:
            violations.append({
                'sigma': float(sigma_test),
                't': float(t),
                'zeta_mag': float(zeta_mag),
                'zeta_val': zeta_val
            })
            print(f"⚠️ Suspicious point at σ={sigma_test:.4f}, t={t:.1f} |ζ|={zeta_mag:.3e}")

    # Generate verdict
    print("\n📊 Scan Results:")
    print(f"Total points scanned: {len(t_values)}")
    print(f"Minimum |ζ(s)| encountered: {min_zeta_mag:.3e} at σ={min_zeta_loc[0]:.6f}, t={min_zeta_loc[1]:.1f}")
    print(f"Potential violations found: {len(violations)}")

    if not violations:
        print("\n✅ No violations: shell integrity confirmed.")
        print(f"All |ζ(s)| > {epsilon:.1e} in scanned region")
    else:
        print("\n⚠️ Breach detected in the critical shell!")
        print(f"{len(violations)} points with |ζ(s)| < {epsilon:.1e}")

    # Create heatmap visualization
    print("\n🎨 Generating heatmap...")
    sigma_grid = np.linspace(0.7, 1.0, 100)
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), 50)
    zeta_mags = np.zeros((len(sigma_grid), len(t_grid)))

    for i, sigma in enumerate(sigma_grid):
        for j, t in enumerate(t_grid):
            s = mpmath.mpc(sigma, t)
            zeta_mags[i,j] = float(abs(mpmath.zeta(s)))

    plt.figure(figsize=(12, 6))

    # Heatmap
    plt.subplot(1, 2, 1)
    X, Y = np.meshgrid(t_grid, sigma_grid)
    plt.pcolormesh(X, Y, np.log10(zeta_mags), shading='auto', cmap='viridis')
    plt.colorbar(label='log10|ζ(s)|')

    # Plot zero-free boundary
    boundary_curve = [1 - c / np.log(t) for t in t_grid]
    plt.plot(t_grid, boundary_curve, 'r-', linewidth=2, label='Theoretical boundary')

    plt.xscale('log')
    plt.xlabel('t (imaginary part)')
    plt.ylabel('σ (real part)')
    plt.title('|ζ(s)| in Critical Strip')
    plt.legend()

    # Violation markers
    if violations:
        viol_t = [v['t'] for v in violations]
        viol_sigma = [v['sigma'] for v in violations]
        plt.scatter(viol_t, viol_sigma, c='red', s=50, marker='x', label='Potential violations')

    # Boundary zoom
    plt.subplot(1, 2, 2)
    boundary_plus = [1 - (c - 0.05) / np.log(t) for t in t_grid]
    boundary_minus = [1 - (c + 0.05) / np.log(t) for t in t_grid]

    plt.fill_between(t_grid, boundary_minus, boundary_plus, color='red', alpha=0.2, label='Scan region')
    plt.plot(t_grid, boundary_curve, 'r-', linewidth=2)

    if violations:
        plt.scatter(viol_t, viol_sigma, c='red', s=50, marker='x')

    plt.xscale('log')
    plt.xlabel('t (imaginary part)')
    plt.ylabel('σ (real part)')
    plt.title('Zero-Free Boundary Detail')
    plt.legend()

    plt.tight_layout()
    plt.savefig('critical_shell.png')
    print("\n🌌 Scan complete. The Riemann frontier holds... for now.")

if __name__ == "__main__":
    main()
