import 'package:flutter_test/flutter_test.dart';
import 'package:dft_monitor_flutter/src/structures/bonds.dart';
import 'package:dft_monitor_flutter/src/structures/cif_parser.dart';

/// CsPbI3 en su fase alfa cubica (Pm-3m), a = 6.1834 A.
///
/// Va incrustado y no en un fichero a proposito: la version anterior leia
/// `test/_alpha.cif`, que estaba en .gitignore, asi que el test fallaba en
/// cualquier copia limpia del repositorio con un PathNotFoundException.
const _cspbi3Cubico = '''
data_image0
_chemical_formula_structural       CsPbI3
_chemical_formula_sum              "Cs1 Pb1 I3"
_cell_length_a       6.1834
_cell_length_b       6.1834
_cell_length_c       6.1834
_cell_angle_alpha    90.0
_cell_angle_beta     90.0
_cell_angle_gamma    90.0

_space_group_name_H-M_alt    "P 1"
_space_group_IT_number       1

loop_
  _space_group_symop_operation_xyz
  'x, y, z'

loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
  Cs  Cs1       1.0  0.0  0.0  0.0  1.0000
  Pb  Pb1       1.0  0.5  0.5  0.5  1.0000
  I   I1        1.0  0.5  0.5  0.0  1.0000
  I   I2        1.0  0.0  0.5  0.5  1.0000
  I   I3        1.0  0.5  0.0  0.5  1.0000
''';

void main() {
  test('CsPbI3 cubico: tres enlaces Pb-I a media arista', () {
    final st = parseCif(_cspbi3Cubico);
    expect(st.atoms.length, 5);

    final bonds = detectarEnlaces(st.atoms);

    // Tres, no seis: el octaedro se completa con yodos de las celdas vecinas y
    // aqui solo hay una celda. Los contactos Cs-I quedan fuera por diseno --- son
    // ionicos y a distancia parecida, y dibujarlos convierte la celda en una
    // marana.
    expect(bonds.length, 3);
    for (final b in bonds) {
      final pareja = {st.atoms[b.i].symbol, st.atoms[b.j].symbol};
      expect(pareja, {'Pb', 'I'});
      final d = st.atoms[b.i].position.distanceTo(st.atoms[b.j].position);
      expect(d, closeTo(6.1834 / 2, 1e-3));
    }
  });

  test('el sitio A no se enlaza: Cs queda suelto', () {
    final st = parseCif(_cspbi3Cubico);
    final bonds = detectarEnlaces(st.atoms);
    final conCs = bonds.where(
      (b) => st.atoms[b.i].symbol == 'Cs' || st.atoms[b.j].symbol == 'Cs',
    );
    expect(conCs, isEmpty);
  });
}
