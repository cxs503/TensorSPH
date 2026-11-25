import pytest
import torch

from tensorsph.heat_conduction_2d import (solve_harmonic_layered_slab,
                                          solve_transient_layered_slab, two_column_sheet_materials)


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

@pytest.mark.parametrize("face", [1, 2, 3])
def test_sheet_preserves_contacts_and_steady_bulk_temperature(face):
    dx = .125
    k = torch.tensor([1., 2., .5, 3.], dtype=torch.float64)
    cv = torch.tensor([2., 1., 3., 4.], dtype=k.dtype)
    contacts = torch.tensor([.2, .4, .1], dtype=k.dtype)
    sheet_r = .6
    material = two_column_sheet_materials(k, cv, dx, face, contact_resistance_x=contacts,
                                          resistance=sheet_r, areal_heat_capacity=.8, sheet_conductivity=2.)
    field, cuts = solve_harmonic_layered_slab(torch.zeros(6), dx, (2., .5), 0.,
                                              **material, return_x_flux=True)
    q = 1.5/((dx/k).sum()+contacts.sum()+sheet_r)
    upstream = (dx/k).cumsum(0)-dx/(2*k) + torch.cat((k.new_zeros(1), contacts.cumsum(0)))
    upstream[face:] += sheet_r
    bulk = torch.cat((field[:face], field[face+2:])).real
    torch.testing.assert_close(bulk, 2.-q*upstream, atol=1e-12, rtol=0)
    torch.testing.assert_close(cuts.real, q.expand_as(cuts.real), atol=1e-12, rtol=0)
    torch.testing.assert_close(dx*material["volumetric_heat_capacity"].sum(), dx*cv.sum()+.8)


def test_two_sheets_can_be_inserted_in_either_order():
    args = dict(conductivity=[1.]*5, volumetric_heat_capacity=2., contact_resistance_x=[.2, .1, .6, .3])
    a = dict(resistance=.4, areal_heat_capacity=1., sheet_conductivity=3.)
    b = dict(resistance=.8, areal_heat_capacity=2., sheet_conductivity=4.)
    ab = two_column_sheet_materials(**two_column_sheet_materials(**args, dx=.1, face=1, **a),
                                    dx=.1, face=5, **b)
    ba = two_column_sheet_materials(**two_column_sheet_materials(**args, dx=.1, face=3, **b),
                                    dx=.1, face=1, **a)
    for key in ab:
        torch.testing.assert_close(ab[key], ba[key])


@pytest.mark.parametrize("contacts", [[.1], [0., -.1], [float("inf"), 0.], [0j, .2j]])
def test_invalid_bulk_contacts(contacts):
    with pytest.raises(ValueError):
        two_column_sheet_materials([1.]*3, 1., .125, 1, resistance=.5, areal_heat_capacity=1.,
                                   sheet_conductivity=1., contact_resistance_x=contacts)
