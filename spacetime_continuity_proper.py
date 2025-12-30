#!/usr/bin/env python3
"""
THE CORRECT MATHEMATICS OF SPACETIME CONTINUITY
A differential geometry approach showing what actually makes spacetime continuous

Key insight: Spacetime continuity comes from the SMOOTH STRUCTURE of the manifold,
not from intersecting light cones. The fundamental objects are:
1. Tangent bundle TM → smooth assignment of tangent spaces
2. Connection ∇ → how to parallel transport between points
3. Geodesics → curves that "follow" the connection
4. Exponential map → how to reach nearby points along geodesics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):
    """Helper class for 3D arrows"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def christoffel_symbols_flat():
    """
    In flat Minkowski spacetime, Christoffel symbols vanish
    But let's compute them to show the machinery
    """
    doc = """
╔══════════════════════════════════════════════════════════════════════╗
║  1. CHRISTOFFEL SYMBOLS: THE CONNECTION COEFFICIENTS                 ║
╚══════════════════════════════════════════════════════════════════════╝

The connection ∇ is defined by Christoffel symbols Γ^λ_μν:

    Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)

These tell you HOW VECTORS CHANGE as you move from point to point.

FLAT SPACETIME (Minkowski):
────────────────────────────
Metric: g_μν = diag(1, -1, -1, -1)  (signature +---)

Since g_μν is constant everywhere:
    ∂_μ g_νσ = 0  for all μ, ν, σ

Therefore:
    Γ^λ_μν = 0  everywhere

This means: Parallel transport preserves vectors!
A vector keeps pointing "the same way" as you move it around.

CURVED SPACETIME (e.g., Schwarzschild):
───────────────────────────────────────
For the Schwarzschild metric (around a mass M):

    ds² = (1 - 2M/r)dt² - (1 - 2M/r)^(-1)dr² - r²dθ² - r²sin²θ dφ²

Non-zero Christoffel symbols include:

    Γ^t_tr = M/[r²(1 - 2M/r)]
    Γ^r_tt = M(1 - 2M/r)/r²
    Γ^r_rr = -M/[r²(1 - 2M/r)]
    Γ^r_θθ = -(r - 2M)
    Γ^θ_rθ = 1/r
    ... (many more)

These NON-ZERO values mean spacetime is CURVED.
Parallel transport now CHANGES vectors as you move them!

KEY INSIGHT:
───────────
The connection Γ^λ_μν is what LITERALLY CONNECTS neighboring points.
It's the answer to: "How does my tangent space at p relate to 
                     the tangent space at p + dp?"
"""
    return doc

def parallel_transport_visualization():
    """
    Visualize parallel transport on a curved surface
    Shows how a vector changes as you transport it
    """
    fig = plt.figure(figsize=(16, 6))
    
    # ============ LEFT: Flat space (trivial parallel transport) ============
    ax1 = fig.add_subplot(131)
    
    # Path in flat space
    t = np.linspace(0, 2*np.pi, 50)
    x_path = np.cos(t)
    y_path = np.sin(t)
    
    ax1.plot(x_path, y_path, 'b-', linewidth=2, label='Closed path')
    ax1.plot([x_path[0]], [y_path[0]], 'go', markersize=10, label='Start/End')
    
    # Vector at various points (stays parallel in flat space)
    sample_points = [0, 10, 20, 30, 40]
    for i in sample_points:
        ax1.arrow(x_path[i], y_path[i], 0.3, 0, 
                 head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.7)
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('FLAT SPACE\nParallel Transport (Γ = 0)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.text(0, -1.9, 'Vector returns to original orientation\nΓ^λ_μν = 0 everywhere',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ MIDDLE: Sphere (non-trivial parallel transport) ============
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Sphere surface
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='lightblue')
    
    # Path on sphere (triangle on equator + meridian)
    path_t = np.linspace(0, np.pi/2, 20)
    
    # Segment 1: Along equator
    x1 = np.cos(path_t)
    y1 = np.sin(path_t)
    z1 = np.zeros_like(path_t)
    
    # Segment 2: Up meridian
    x2 = np.zeros(20)
    y2 = np.ones(20)
    z2 = np.linspace(0, 1, 20)
    
    # Segment 3: Back along meridian
    x3 = np.linspace(0, 1, 20)
    y3 = np.zeros(20)
    z3 = np.linspace(1, 0, 20)
    
    ax2.plot(x1, y1, z1, 'r-', linewidth=3, label='Path on sphere')
    ax2.plot(x2, y2, z2, 'r-', linewidth=3)
    ax2.plot(x3, y3, z3, 'r-', linewidth=3)
    
    ax2.scatter([1], [0], [0], color='green', s=100, label='Start')
    ax2.scatter([1], [0], [0], color='red', s=100, marker='x', label='End (rotated!)')
    
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('SPHERE (Curved)\nHolonomy Effect', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # ============ RIGHT: Connection visualization ============
    ax3 = fig.add_subplot(133)
    
    # Grid of points representing spacetime
    x_grid = np.arange(-2, 3, 1)
    y_grid = np.arange(-2, 3, 1)
    
    for x in x_grid:
        for y in y_grid:
            ax3.plot(x, y, 'ko', markersize=6)
            
            # Draw tangent space as small cross
            ax3.plot([x-0.15, x+0.15], [y, y], 'gray', linewidth=0.5, alpha=0.5)
            ax3.plot([x, x], [y-0.15, y+0.15], 'gray', linewidth=0.5, alpha=0.5)
    
    # Show connection linking neighboring spaces
    # Arrows showing how vectors at (0,0) relate to vectors at neighbors
    center_x, center_y = 0, 0
    neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for dx, dy in neighbors:
        # Connection arrow
        ax3.annotate('', xy=(center_x + dx, center_y + dy), 
                    xytext=(center_x, center_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
        
        # Show vector transformation
        ax3.arrow(center_x + dx, center_y + dy, 0.3, 0.3,
                 head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.5)
    
    # Highlight center point
    ax3.plot(center_x, center_y, 'ro', markersize=12)
    ax3.arrow(center_x, center_y, 0.3, 0.3,
             head_width=0.15, head_length=0.08, fc='red', ec='red', linewidth=2)
    
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('CONNECTION ∇\nLinking Tangent Spaces', fontsize=12, fontweight='bold')
    ax3.text(0, -3, 'Blue arrows: connection Γ^λ_μν\nRed arrows: vectors in tangent spaces\n' +
             'Connection tells how red vectors relate at different points',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/parallel_transport.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: parallel_transport.png")
    plt.close()

def geodesic_flow_and_exponential_map():
    """
    Show how geodesics flow out from a point and the exponential map
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ============ LEFT: Geodesic spray from a point ============
    
    # Point p in spacetime
    p_x, p_y = 0, 0
    
    ax1.plot(p_x, p_y, 'ro', markersize=15, label='Point p', zorder=5)
    
    # Draw tangent space at p
    ax1.arrow(p_x, p_y, 1, 0, head_width=0.1, head_length=0.05, 
             fc='gray', ec='gray', alpha=0.3, linewidth=2)
    ax1.arrow(p_x, p_y, 0, 1, head_width=0.1, head_length=0.05, 
             fc='gray', ec='gray', alpha=0.3, linewidth=2)
    ax1.text(0.5, -0.3, 'Tangent space T_p M', fontsize=10, style='italic')
    
    # Geodesics in various directions (in flat space, these are straight lines)
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    for angle in angles:
        # Initial velocity vector
        v_x = np.cos(angle)
        v_y = np.sin(angle)
        
        # Geodesic: γ(t) = p + t*v (in flat space)
        t = np.linspace(0, 2, 50)
        gamma_x = p_x + t * v_x
        gamma_y = p_y + t * v_y
        
        ax1.plot(gamma_x, gamma_y, 'b-', alpha=0.6, linewidth=1.5)
        
        # Mark endpoint at t=1
        ax1.plot(gamma_x[25], gamma_y[25], 'go', markersize=4, alpha=0.7)
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('GEODESIC FLOW from point p\n' + 
                 'Each tangent vector v ∈ T_p M generates a geodesic γ_v(t)',
                 fontsize=12, fontweight='bold')
    
    ax1.text(0, -3.2, 'Geodesic equation: d²γ^μ/dt² + Γ^μ_αβ (dγ^α/dt)(dγ^β/dt) = 0\n' +
            'In flat space: Γ = 0, so geodesics are straight lines',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ RIGHT: Exponential map exp_p : T_p M → M ============
    
    ax2.plot(p_x, p_y, 'ro', markersize=15, label='Point p', zorder=5)
    
    # Draw multiple tangent vectors and their exponential map images
    vectors = [
        (1.5, 0, 'Vector v₁'),
        (0, 1.5, 'Vector v₂'),
        (-1.2, 0.8, 'Vector v₃'),
        (1, 1, 'Vector v₄'),
    ]
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    for i, (v_x, v_y, label) in enumerate(vectors):
        # Draw vector in tangent space
        ax2.arrow(p_x, p_y, v_x, v_y, 
                 head_width=0.15, head_length=0.1, 
                 fc=colors[i], ec=colors[i], alpha=0.7, linewidth=2)
        
        # Draw geodesic to exp_p(v)
        t = np.linspace(0, 1, 50)
        gamma_x = p_x + t * v_x
        gamma_y = p_y + t * v_y
        ax2.plot(gamma_x, gamma_y, '--', color=colors[i], alpha=0.5, linewidth=1.5)
        
        # Mark the exponential map image exp_p(v)
        ax2.plot(p_x + v_x, p_y + v_y, 'o', color=colors[i], 
                markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax2.text(p_x + v_x + 0.2, p_y + v_y + 0.2, f'exp_p(v{i+1})', 
                fontsize=9, color=colors[i], fontweight='bold')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('EXPONENTIAL MAP: exp_p : T_p M → M\n' + 
                 'Maps tangent vectors to points via geodesics',
                 fontsize=12, fontweight='bold')
    
    ax2.text(0, -3.2, 'exp_p(v) = γ_v(1) = "follow geodesic with velocity v for time 1"\n' +
            'This is how neighboring points are SMOOTHLY connected!',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/geodesic_exponential.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: geodesic_exponential.png")
    plt.close()

def tangent_bundle_structure():
    """
    Visualize the tangent bundle TM as fiber bundle structure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # ============ TOP: The bundle structure ============
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Base manifold M (2D for visualization)
    theta = np.linspace(0, 2*np.pi, 30)
    r = np.linspace(0.5, 2, 20)
    Theta, R = np.meshgrid(theta, r)
    X_base = R * np.cos(Theta)
    Y_base = R * np.sin(Theta)
    Z_base = np.zeros_like(X_base)
    
    ax1.plot_surface(X_base, Y_base, Z_base, alpha=0.3, color='lightblue', 
                     edgecolor='none', label='Base manifold M')
    
    # Several points with their tangent spaces (fibers)
    points = [
        (1, 0, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0.7, 0.7, 0),
    ]
    
    for px, py, pz in points:
        # Point on base
        ax1.scatter([px], [py], [pz], color='red', s=50, zorder=5)
        
        # Tangent space fiber (vertical plane above point)
        fiber_x = px + np.linspace(-0.3, 0.3, 10)
        fiber_y = py + np.linspace(-0.3, 0.3, 10)
        FX, FY = np.meshgrid(fiber_x, fiber_y)
        FZ = np.ones_like(FX) * 1.5
        
        ax1.plot_surface(FX, FY, FZ, alpha=0.2, color='yellow')
        
        # Some vectors in the tangent space
        for _ in range(3):
            v_len = 0.3
            v_x = px + np.random.uniform(-0.2, 0.2)
            v_y = py + np.random.uniform(-0.2, 0.2)
            
            # Vector shown as vertical arrow in fiber
            ax1.plot([px, v_x], [py, v_y], [1.5, 1.5], 
                    'g-', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Fiber direction')
    ax1.set_title('TANGENT BUNDLE TM\nFiber Bundle Structure', 
                 fontsize=12, fontweight='bold')
    ax1.text2D(0.5, 0.02, 'Each point p ∈ M has tangent space T_p M as fiber above it',
              transform=ax1.transAxes, ha='center', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ TOP RIGHT: Smooth section (vector field) ============
    ax2 = fig.add_subplot(222)
    
    # Grid of points on manifold
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # Smooth vector field (a section of TM)
    # Example: rotating vector field
    U = -Y
    V = X
    
    ax2.quiver(X, Y, U, V, alpha=0.7, scale=30, width=0.004, color='blue')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('SMOOTH SECTION (Vector Field)\ns: M → TM', 
                 fontsize=12, fontweight='bold')
    ax2.text(0, -3.2, 'A section assigns a vector v_p ∈ T_p M to each point p\n' +
            'Smoothness of s is what makes this a CONTINUOUS field',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ BOTTOM LEFT: Local trivialization ============
    ax3 = fig.add_subplot(223)
    
    # Show that locally TM looks like M × ℝⁿ
    # Draw a patch on M
    patch_x = np.linspace(-1, 1, 20)
    patch_y = np.linspace(-1, 1, 20)
    PX, PY = np.meshgrid(patch_x, patch_y)
    
    ax3.contourf(PX, PY, np.zeros_like(PX), levels=1, colors='lightblue', alpha=0.5)
    ax3.contour(PX, PY, np.zeros_like(PX), levels=1, colors='blue', linewidths=2)
    
    # Show several points with their tangent vectors
    for px in [-0.5, 0, 0.5]:
        for py in [-0.5, 0, 0.5]:
            ax3.plot(px, py, 'ro', markersize=8)
            # Draw a few tangent vectors
            ax3.arrow(px, py, 0.2, 0.1, head_width=0.05, head_length=0.03, 
                     fc='green', ec='green', alpha=0.6)
            ax3.arrow(px, py, -0.1, 0.2, head_width=0.05, head_length=0.03, 
                     fc='purple', ec='purple', alpha=0.6)
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x (base M)', fontsize=12)
    ax3.set_ylabel('y (base M)', fontsize=12)
    ax3.set_title('LOCAL TRIVIALIZATION\nπ^(-1)(U) ≅ U × ℝⁿ', 
                 fontsize=12, fontweight='bold')
    ax3.text(0, -1.9, 'Locally, tangent bundle looks like product:\n' +
            'neighborhood U × all possible vectors',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ BOTTOM RIGHT: Coordinate chart explanation ============
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    explanation = """
╔════════════════════════════════════════════════════════════════╗
║           TANGENT BUNDLE: THE COMPLETE STRUCTURE               ║
╚════════════════════════════════════════════════════════════════╝

DEFINITION:
───────────
TM = ⋃ T_p M  (disjoint union of all tangent spaces)
    p∈M

With projection π : TM → M that sends vectors to their base points.

SMOOTH STRUCTURE:
─────────────────
For chart (U, φ) on M with φ: U → ℝⁿ, we get chart on TM:

    Φ: π^(-1)(U) → ℝⁿ × ℝⁿ
    Φ(v_p) = (φ(p), Dφ_p(v))

This makes TM into a 2n-dimensional smooth manifold!

SECTIONS (Vector Fields):
─────────────────────────
A smooth section s: M → TM satisfies π ∘ s = id_M

In coordinates: s(p) = (p, X^μ(p))

Smoothness means X^μ(p) are smooth functions.
This is what it means for a vector field to be "continuous"!

CONNECTION:
───────────
A connection ∇ defines covariant derivative:

    ∇_X Y = X^μ ∂_μ Y^ν + X^μ Y^σ Γ^ν_μσ

This is how we "differentiate vector fields" - how we compare
vectors at different points!

THE KEY INSIGHT:
───────────────
Spacetime continuity = Smooth tangent bundle + Connection

NOT: Intersection of light cones!

The connection Γ^λ_μν literally tells you how to SMOOTHLY
transport vectors from T_p M to T_q M as you move from p to q.

THIS is what "contiguous linking" actually means!
    """
    
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/tangent_bundle_structure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tangent_bundle_structure.png")
    plt.close()

def curvature_and_holonomy():
    """
    Show how curvature (Riemann tensor) emerges from parallel transport around loops
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ============ LEFT: Parallel transport around a small loop ============
    
    # Small parallelogram in spacetime
    p_x, p_y = 0, 0
    
    # Two basis vectors at p
    v1 = np.array([1.0, 0])
    v2 = np.array([0, 1.0])
    
    # Construct parallelogram
    corner_0 = np.array([p_x, p_y])
    corner_1 = corner_0 + 0.5 * v1
    corner_2 = corner_1 + 0.5 * v2
    corner_3 = corner_0 + 0.5 * v2
    
    # Draw parallelogram
    corners = np.array([corner_0, corner_1, corner_2, corner_3, corner_0])
    ax1.plot(corners[:, 0], corners[:, 1], 'b-', linewidth=2)
    ax1.plot(corner_0[0], corner_0[1], 'ro', markersize=12, label='Starting point p')
    
    # Test vector to parallel transport
    test_vec = np.array([0.4, 0.3])
    
    # Show the vector at each corner (in flat space it would be the same)
    # But add small rotation to show curvature effect
    angles = [0, 5, 10, 15, 20]  # degrees - simulating curvature
    for i, corner in enumerate(corners[:-1]):
        angle_rad = np.deg2rad(angles[i])
        rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                              [np.sin(angle_rad), np.cos(angle_rad)]])
        rotated_vec = rot_matrix @ test_vec
        
        ax1.arrow(corner[0], corner[1], rotated_vec[0], rotated_vec[1],
                 head_width=0.08, head_length=0.05, 
                 fc='red', ec='red', alpha=0.7, linewidth=1.5)
    
    # Show the discrepancy at return
    final_angle = np.deg2rad(angles[-1])
    final_rot = np.array([[np.cos(final_angle), -np.sin(final_angle)],
                         [np.sin(final_angle), np.cos(final_angle)]])
    final_vec = final_rot @ test_vec
    
    # Draw both initial and final for comparison
    ax1.arrow(corner_0[0], corner_0[1], test_vec[0], test_vec[1],
             head_width=0.08, head_length=0.05, 
             fc='green', ec='green', alpha=0.5, linewidth=2, 
             linestyle='--', label='Initial vector')
    
    # Arc showing rotation
    arc_angles = np.linspace(0, final_angle, 20)
    arc_r = 0.6
    arc_x = corner_0[0] + arc_r * np.cos(arc_angles)
    arc_y = corner_0[1] + arc_r * np.sin(arc_angles)
    ax1.plot(arc_x, arc_y, 'purple', linewidth=2, label='Holonomy angle')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('HOLONOMY: Parallel Transport Around Loop\n' +
                 'Vector returns ROTATED in curved space',
                 fontsize=12, fontweight='bold')
    
    ax1.text(0.5, -0.8, 'Holonomy = how much vector rotates\n' +
            'Directly related to Riemann curvature tensor R^λ_μνρ',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============ RIGHT: Riemann tensor from commutator ============
    ax2.axis('off')
    
    riemann_doc = """
╔═══════════════════════════════════════════════════════════════════╗
║              CURVATURE: THE RIEMANN TENSOR                        ║
╚═══════════════════════════════════════════════════════════════════╝

DEFINITION via Connection Commutator:
──────────────────────────────────────

For vector field V and coordinate directions ∂_μ, ∂_ν:

    R(∂_μ, ∂_ν)V = ∇_μ ∇_ν V - ∇_ν ∇_μ V

In components:

    R^λ_ρμν = ∂_μ Γ^λ_νρ - ∂_ν Γ^λ_μρ + Γ^λ_μσ Γ^σ_νρ - Γ^λ_νσ Γ^σ_μρ

PHYSICAL MEANING:
─────────────────

R^λ_ρμν measures how much parallel transport around an 
infinitesimal loop (in μ-ν plane) rotates a vector in the λ direction.

Small loop with edges δx^μ and δx^ν:

    ΔV^λ = R^λ_ρμν V^ρ δx^μ δx^ν

If R = 0 everywhere → Flat spacetime (Minkowski)
If R ≠ 0           → Curved spacetime (Gravity!)

RELATIONSHIP TO HOLONOMY:
─────────────────────────

For parallelogram with sides V, W:

    Holonomy angle ≈ R(V, W) · area

This is GEOMETRIC. Curvature = failure of parallel transport
to preserve vectors around closed loops.

EINSTEIN'S EQUATIONS:
────────────────────

    G_μν = R_μν - (1/2)R g_μν = (8πG/c⁴) T_μν

    where R_μν = R^λ_μλν  (Ricci tensor)
          R = g^μν R_μν    (Ricci scalar)

Matter/energy (T_μν) tells spacetime how to curve!

THE DEEP INSIGHT:
────────────────

Curvature ISN'T just a property of the space.
It's a property of the CONNECTION ∇.

The connection is what links neighboring points.
Curvature measures the failure of this linking to be "flat."

FLAT:   R = 0  →  Γ = 0 in some coordinates
CURVED: R ≠ 0  →  Cannot make Γ = 0 everywhere

This is what makes spacetime GENUINELY curved vs just
using curvy coordinates on flat space!
    """
    
    ax2.text(0.05, 0.98, riemann_doc, transform=ax2.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/curvature_holonomy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: curvature_holonomy.png")
    plt.close()

def generate_master_document():
    """
    Generate comprehensive LaTeX-style document explaining everything
    """
    doc = """
═══════════════════════════════════════════════════════════════════════
         THE CORRECT MATHEMATICS OF SPACETIME CONTINUITY
           A Differential Geometry Foundation
═══════════════════════════════════════════════════════════════════════

                        by Claude & Steve
                        December 30, 2025


ABSTRACT
════════

This document presents the rigorous mathematical foundation for 
understanding spacetime continuity. We show that the continuum emerges 
from the smooth structure of the tangent bundle, the connection (covariant 
derivative), and geodesic flow, NOT from intersecting light cones. Light 
cones are derived objects that encode causality but do not explain 
continuity.


1. THE MANIFOLD STRUCTURE
═══════════════════════════════════════════════════════════════════════

1.1 DEFINITION: SMOOTH MANIFOLD
────────────────────────────────

A smooth manifold M is a topological space with:

a) Hausdorff property: Distinct points have disjoint neighborhoods
b) Second-countable: Has countable basis of open sets  
c) Locally Euclidean: Each point has neighborhood ≅ ℝⁿ
d) Smooth atlas: Charts (U_α, φ_α) with smooth transition maps

For spacetime: M is 4-dimensional with signature (+,-,-,-)


1.2 WHAT CONTINUITY MEANS
──────────────────────────

"Continuous" ≠ just topologically connected!

Spacetime continuity requires:

1. SMOOTH functions: C^∞ maps between manifolds
2. SMOOTH curves: γ: ℝ → M with smooth components
3. SMOOTH tensor fields: g_μν, Γ^λ_μν, etc.

Mathematical criterion: All coordinate transition maps are C^∞

This is MUCH stronger than just "no gaps."


2. THE TANGENT BUNDLE TM
═══════════════════════════════════════════════════════════════════════

2.1 DEFINITION
──────────────

The tangent bundle is:

    TM = ⋃ {p} × T_p M
        p∈M

where T_p M is the tangent space at p (all tangent vectors at p).

Projection: π: TM → M  sends (p, v) ↦ p


2.2 SMOOTH STRUCTURE ON TM
───────────────────────────

TM is itself a smooth manifold of dimension 2n (where dim M = n).

Given chart (U, φ) on M, induced chart on TM:

    Φ: π^(-1)(U) → ℝⁿ × ℝⁿ
    Φ(p, v) = (φ(p), Dφ_p(v))

Smooth structure on TM makes vector fields smooth sections.


2.3 VECTOR FIELDS AS SMOOTH SECTIONS
─────────────────────────────────────

A vector field X is a section s: M → TM with π ∘ s = id_M

In coordinates: X(p) = X^μ(p) ∂_μ|_p

X is smooth ⟺ X^μ are smooth functions

THIS is what it means for a vector field to vary continuously!


3. THE CONNECTION: LINKING TANGENT SPACES
═══════════════════════════════════════════════════════════════════════

3.1 THE PROBLEM
───────────────

Given:
  - Vector v at point p  
  - Vector w at nearby point q

Question: How do we compare them?

We can't just subtract: v ∈ T_p M, w ∈ T_q M (different spaces!)

Solution: PARALLEL TRANSPORT via connection ∇


3.2 COVARIANT DERIVATIVE
─────────────────────────

A connection ∇ assigns to each pair of vector fields (X, Y) a new 
vector field ∇_X Y satisfying:

1. ∇_{fX+gY} Z = f∇_X Z + g∇_Y Z     (linearity in X)
2. ∇_X(Y + Z) = ∇_X Y + ∇_X Z        (additivity)
3. ∇_X(fY) = (Xf)Y + f∇_X Y          (Leibniz rule)

In coordinates:

    ∇_X Y = X^μ(∂_μ Y^ν + Γ^ν_μρ Y^ρ) ∂_ν

The Christoffel symbols Γ^λ_μν ARE the connection coefficients.


3.3 CHRISTOFFEL SYMBOLS
────────────────────────

For metric connection (Levi-Civita), Γ^λ_μν is symmetric and:

    Γ^λ_μν = (1/2) g^λσ(∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)

PHYSICAL MEANING:

Γ^λ_μν tells you how the λ-component of a vector changes when you 
parallel transport it in the μ-direction along a curve whose tangent 
points in the ν-direction.

If Γ = 0 everywhere (in some coordinates): FLAT spacetime
If Γ ≠ 0 but can't be made 0 everywhere: CURVED spacetime


3.4 PARALLEL TRANSPORT
───────────────────────

Transport vector V along curve γ(t) by solving:

    ∇_{γ'} V = 0

In components:

    dV^μ/dt + Γ^μ_αβ V^α dγ^β/dt = 0

This ODE has unique solution. Parallel transport:
  - Preserves vector lengths: g(V, V) = constant
  - Is path-dependent in curved space!


4. GEODESICS: THE FABRIC OF SPACETIME
═══════════════════════════════════════════════════════════════════════

4.1 GEODESIC EQUATION
──────────────────────

A curve γ(t) is a geodesic if it parallel transports its own tangent:

    ∇_{γ'} γ' = 0

In coordinates:

    d²γ^μ/dt² + Γ^μ_αβ (dγ^α/dt)(dγ^β/dt) = 0

This is the "straightest possible" curve in curved spacetime.


4.2 EXISTENCE AND UNIQUENESS
─────────────────────────────

THEOREM (Geodesic completeness):

For any point p ∈ M and tangent vector v ∈ T_p M, there exists 
unique maximal geodesic γ_v with:

    γ_v(0) = p
    γ'_v(0) = v

This can be extended to all t (for complete manifolds).


4.3 GEODESIC FLOW
──────────────────

Define map: Φ^t: TM → TM that flows along geodesics:

    Φ^t(p, v) = (γ_v(t), γ'_v(t))

This is a SMOOTH flow on the tangent bundle!

Geometric picture: Geodesics spray out smoothly from each point in 
all directions. THIS creates the continuous structure.


5. EXPONENTIAL MAP: THE LINKING MECHANISM
═══════════════════════════════════════════════════════════════════════

5.1 DEFINITION
──────────────

For point p, define exponential map:

    exp_p: T_p M → M
    exp_p(v) = γ_v(1)

"Follow geodesic with initial velocity v for parameter time 1."


5.2 PROPERTIES
──────────────

1. exp_p(0) = p
2. D(exp_p)_0 = identity map
3. exp_p is diffeomorphism from neighborhood of 0 ∈ T_p M to 
   neighborhood of p ∈ M (inverse function theorem)
4. exp_p(tv) = γ_v(t) for all t


5.3 THE KEY INSIGHT
───────────────────

The exponential map is how we reach nearby points!

Given point p and "direction" v ∈ T_p M:
→ exp_p(v) is the nearby point you reach

THIS is the mathematical meaning of "contiguous linking":

    Points near p ⟷ Vectors near 0 ∈ T_p M
    via smooth map exp_p

The SMOOTHNESS of exp_p is what makes spacetime continuous!


5.4 NORMAL COORDINATES
───────────────────────

Using exp_p, define coordinates around p:

    x^μ = coordinates of v ∈ T_p M
    ↓ exp_p
    point q = exp_p(v)

In these coordinates:
  - g_μν(p) = η_μν (Minkowski metric at p)
  - Γ^λ_μν(p) = 0 (connection vanishes at p)
  - ∂_σ g_μν(p) = 0 (first derivatives vanish)

But Γ^λ_μν ≠ 0 away from p if there's curvature!


6. CURVATURE: OBSTRUCTION TO FLATNESS
═══════════════════════════════════════════════════════════════════════

6.1 RIEMANN CURVATURE TENSOR
─────────────────────────────

Define via commutator of covariant derivatives:

    R(X, Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z - ∇_{[X,Y]} Z

In components:

    R^λ_ρμν = ∂_μ Γ^λ_νρ - ∂_ν Γ^λ_μρ 
              + Γ^λ_μσ Γ^σ_νρ - Γ^λ_νσ Γ^σ_μρ


6.2 GEOMETRIC MEANING
──────────────────────

R^λ_ρμν measures parallel transport around infinitesimal loop.

For small parallelogram with sides δx^μ, δx^ν:

    ΔV^λ = R^λ_ρμν V^ρ δx^μ δx^ν

If R = 0 everywhere: Spacetime is FLAT (Minkowski)
If R ≠ 0: Spacetime is CURVED (Gravity present)


6.3 EINSTEIN FIELD EQUATIONS
─────────────────────────────

Curvature is determined by matter/energy:

    G_μν = (8πG/c⁴) T_μν

where G_μν = R_μν - (1/2)R g_μν (Einstein tensor)

This tells us HOW the connection Γ^λ_μν varies in space!


7. WHY LIGHT CONES DON'T EXPLAIN CONTINUITY
═══════════════════════════════════════════════════════════════════════

7.1 LIGHT CONES ARE DERIVED
────────────────────────────

Given metric g_μν, light cone at p is:

    {v ∈ T_p M | g_μν v^μ v^ν = 0}

This is DERIVED from the metric. The metric itself comes from the 
smooth structure + Einstein equations.

Light cones don't create continuity - they're a CONSEQUENCE of it.


7.2 THEY EMPHASIZE WRONG THING
───────────────────────────────

Light cones naturally make you think about:
  - Discrete events
  - Causal relationships
  - Intersection regions

But continuity is about:
  - Smooth variation of geometric structures
  - Differential equations (geodesic, Einstein)
  - Local-to-global properties (parallel transport)


7.3 THE CORRECT PICTURE
────────────────────────

Spacetime continuum =
    Smooth manifold M
  + Metric tensor field g_μν (smooth)
  + Connection ∇ (determined by g)
  + Geodesic flow (smooth)
  + Exponential map (smooth)

Light cones = {null vectors at each point}

They're important for CAUSALITY but irrelevant for CONTINUITY.


8. SUMMARY: WHAT CREATES THE CONTINUUM
═══════════════════════════════════════════════════════════════════════

INGREDIENTS:
────────────

1. Smooth manifold M (topology + smooth atlas)
2. Metric g_μν(x) ∈ C^∞ (smooth tensor field)
3. Connection Γ^λ_μν derived from g
4. Geodesic flow Φ^t on TM (smooth)
5. Exponential map exp_p: T_p M → M (smooth)

THE LINKAGE:
────────────

Points p and q are "contiguously linked" via:

  1. Geodesic γ connecting them (if they're not too far)
  2. Parallel transport along γ relating T_p M to T_q M
  3. Exponential map: q = exp_p(v) for some v ∈ T_p M

The SMOOTHNESS of these structures = CONTINUITY of spacetime


THE DEEP TRUTH:
───────────────

Spacetime isn't just a set of points. It's a SMOOTH MANIFOLD with:

  - Smooth curves
  - Smooth tensor fields  
  - Smooth connections
  - Smooth geodesic flow

This smoothness - encoded in C^∞ transition functions, differentiable 
tensor fields, and smooth exponential maps - IS what makes spacetime 
continuous.

Light cones are just one tensor (the null cone of g_μν). They tell us 
about causality, not about the fundamental smooth structure.


FINAL INSIGHT:
──────────────

The question "How do light cones link to form a continuum?" is 
backwards.

The correct question: "How does the smooth manifold structure + 
connection create a continuum in which light cones can be defined?"

Answer: Through the tangent bundle, covariant derivative, geodesic 
flow, and exponential map - all of which are SMOOTH structures that 
literally connect neighboring points.

═══════════════════════════════════════════════════════════════════════
                            END OF DOCUMENT
═══════════════════════════════════════════════════════════════════════
"""
    
    with open('/mnt/user-data/outputs/spacetime_continuity_complete.txt', 'w') as f:
        f.write(doc)
    
    print("\n" + "="*75)
    print("DOCUMENT: THE CORRECT MATHEMATICS OF SPACETIME CONTINUITY")
    print("="*75)
    print(doc)
    print("✓ Saved: spacetime_continuity_complete.txt")

def main():
    """Execute all visualizations and generate documentation"""
    print("\n" + "="*75)
    print("  THE CORRECT MATHEMATICS OF SPACETIME CONTINUITY")
    print("  (Not light cones - the real differential geometry)")
    print("="*75 + "\n")
    
    print("Generating visualizations...\n")
    
    print(christoffel_symbols_flat())
    print("\n")
    
    parallel_transport_visualization()
    geodesic_flow_and_exponential_map()
    tangent_bundle_structure()
    curvature_and_holonomy()
    
    print("\n" + "="*75)
    generate_master_document()
    
    print("\n" + "="*75)
    print("✓ ALL VISUALIZATIONS AND DOCUMENTATION COMPLETE!")
    print("="*75)
    print("\nGenerated files:")
    print("  1. parallel_transport.png - Shows how connection works")
    print("  2. geodesic_exponential.png - Geodesic flow and exp map")
    print("  3. tangent_bundle_structure.png - The fiber bundle TM")
    print("  4. curvature_holonomy.png - Riemann tensor explained")
    print("  5. spacetime_continuity_complete.txt - Complete theory")
    print("\nThese explain what ACTUALLY makes spacetime continuous!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
