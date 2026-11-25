import pytest
import torch

from tensorsph.heat_conduction_2d import solve_transient_layered_slab, two_column_sheet_materials


def test_sheet_materials_keep_bulk_order_and_integrated_properties():
    material = two_column_sheet_materials([1., 2., 3.], [4., 5., 6.], .1, 1,
                                          resistance=.5, areal_heat_capacity=.8, sheet_conductivity=2.)
    assert material["conductivity"].tolist() == [1., 2., 2., 2., 3.]
    assert material["volumetric_heat_capacity"].tolist() == [4., 4., 4., 5., 6.]
    assert material["contact_resistance_x"].tolist() == pytest.approx([.2, 0., .2, 0.])


def test_sheet_power_balance_at_default_kernel_support():
    dx, capacity, face = .125, .75, 3
    material = two_column_sheet_materials(torch.ones(8), 1., dx, face, resistance=.5,
                                          areal_heat_capacity=capacity, sheet_conductivity=8.)
    source = torch.zeros(10, dtype=torch.float64)
    source[face:face+2] = 1/(2*dx)
    times = torch.tensor([0., .004, .011], dtype=torch.float64)
    states, cuts = solve_transient_layered_slab(
        torch.zeros_like(source), times, dx, (.1, .4), **material, heat_source=source, return_x_heat=True)
    stored = capacity*states[:, face:face+2].mean(dim=1)
    torch.testing.assert_close(stored.diff(), times.diff()+cuts[:, face]-cuts[:, face+2],
                               atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("face,resistance", [(0, 1.), (4, 1.), (2, .01)])
def test_invalid_sheet_geometry(face, resistance):
    with pytest.raises(ValueError):
        two_column_sheet_materials([1.]*4, 1., .125, face, resistance=resistance,
                                   areal_heat_capacity=1., sheet_conductivity=1.)
