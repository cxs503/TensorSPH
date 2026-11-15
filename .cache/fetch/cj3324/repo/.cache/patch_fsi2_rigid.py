import pathlib
p = pathlib.Path("src/tensorsph/fsi2_turek_hron.py")
s = p.read_text()

old = '''        if self.beam_length <= 0 or self.beam_thickness <= 0:
            raise ValueError("beam dimensions must be positive")
        if self.n_beam_nodes < 3:
            raise ValueError("beam needs at least 3 nodes")'''
new = '''        if self.beam_length < 0 or self.beam_thickness <= 0:
            raise ValueError("beam dimensions must be non-negative")
        if self.has_beam() and self.n_beam_nodes < 3:
            raise ValueError("beam needs at least 3 nodes")'''
assert old in s
s = s.replace(old, new)

old = '''    def dt(self) -> float:
        return self.max_courant * self.h() / self.sound_speed'''
new = '''    def dt(self) -> float:
        return self.max_courant * self.h() / self.sound_speed

    def has_beam(self) -> bool:
        """``False`` for a rigid-body-only set-up (``beam_length == 0``).

        The Turek-Hron case is posed for the elastic beam, but the *load*
        operator has to be verifiable independently of the structural model:
        with no beam the body reduces to the rigid cylinder of the DFG
        (Schaefer-Turek) laminar-cylinder benchmark, which has an external
        reference.  Nothing else in the case is special-cased -- the cylinder
        surface, the inlet profile and the through-flow boundaries are
        unchanged.
        """
        return self.beam_length > 0.0 and self.n_beam_nodes >= 2'''
assert old in s
s = s.replace(old, new)

old = '''    # Beam
    n_boundary = max(int(math.floor(case.beam_length / case.dx)), 2)
    beam_interp = interpolation_matrix_beam(case.n_beam_nodes, n_boundary, device)
    nodes = beam_initial_nodes(case, device)'''
new = '''    # Beam (absent in the rigid-body-only configuration)
    if case.has_beam():
        n_boundary = max(int(math.floor(case.beam_length / case.dx)), 2)
        beam_interp = interpolation_matrix_beam(case.n_beam_nodes, n_boundary, device)
        nodes = beam_initial_nodes(case, device)
    else:
        beam_interp = torch.zeros((0, 0), dtype=torch.float64, device=device)
        nodes = torch.zeros((0, 2), dtype=torch.float64, device=device)'''
assert old in s
s = s.replace(old, new)

old = '''    # Beam boundary layer: one row per face, outward normals per particle
    beam_boundary, beam_normals, beam_interp, tributary = beam_boundary_geometry(
        state.beam_nodes, case)'''
new = '''    # Beam boundary layer: one row per face, outward normals per particle.
    # With no elastic beam the body is the rigid cylinder alone and these rows
    # are empty; every consumer below handles a zero-row block.
    if state.beam_nodes.shape[0] > 0:
        beam_boundary, beam_normals, beam_interp, tributary = beam_boundary_geometry(
            state.beam_nodes, case)
    else:
        beam_boundary = torch.zeros((0, 2), dtype=state.fluid_pos.dtype,
                                    device=state.fluid_pos.device)
        beam_normals = torch.zeros((0, 2), dtype=state.fluid_pos.dtype,
                                   device=state.fluid_pos.device)
        beam_interp = state.beam_interp
        tributary = 0.0'''
assert old in s
s = s.replace(old, new)

old = '''    # Force on the beam from the fluid: each face carries its own normal
    p_wall, w_sum = boundary_pressure_from_fluid('''
new = '''    # Force on the beam from the fluid: each face carries its own normal
    if beam_normals.shape[0] == 0:
        f_beam = torch.zeros((0, 2), dtype=state.fluid_pos.dtype,
                             device=state.fluid_pos.device)
        coverage = 0.0
    else:
      p_wall, w_sum = boundary_pressure_from_fluid('''
assert old in s
s = s.replace(old, new)

old = '''    f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)
    coverage = float((w_sum > 1.0e-30).to(torch.float64).mean()) if p_wall.numel() else 0.0'''
new = '''      f_beam = boundary_force_from_pressure(p_wall, beam_normals, tributary)
      coverage = (float((w_sum > 1.0e-30).to(torch.float64).mean())
                  if p_wall.numel() else 0.0)'''
assert old in s
s = s.replace(old, new)

old = '''    nodes, node_vel = state.beam_nodes, state.beam_vel
    for _ in range(n_sub):
        nodes, node_vel = beam_advance(nodes, node_vel, nodal_load, case, dt_rod)
    state.beam_nodes, state.beam_vel = nodes, node_vel'''
new = '''    if state.beam_nodes.shape[0] > 0:
        nodes, node_vel = state.beam_nodes, state.beam_vel
        for _ in range(n_sub):
            nodes, node_vel = beam_advance(nodes, node_vel, nodal_load, case, dt_rod)
        state.beam_nodes, state.beam_vel = nodes, node_vel'''
assert old in s
s = s.replace(old, new)

old = '''    tip_x = float(state.beam_nodes[-1, 0]) - (case.cylinder_x + case.cylinder_d / 2 + case.beam_length)
    tip_y = float(state.beam_nodes[-1, 1]) - case.cylinder_y'''
new = '''    if state.beam_nodes.shape[0] > 0:
        tip_x = (float(state.beam_nodes[-1, 0])
                 - (case.cylinder_x + case.cylinder_d / 2 + case.beam_length))
        tip_y = float(state.beam_nodes[-1, 1]) - case.cylinder_y
    else:
        tip_x = tip_y = 0.0'''
assert old in s
s = s.replace(old, new)

p.write_text(s)
print("patched")