"""
Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from unittest import TestCase
import numpy as np

from viktor.geometry import GeoPoint, GeoPolyline
from app.functions.ahn import get_geo_polyline_section_length, _get_map_payload, get_ahn_of_polyline, Polyline


class TestAhn(TestCase):
    """
    Test class that contains all tests for verifying the correct operation of the AHN functions
    """

    def test_get_geo_polyline_length(self):
        """
        Check that a triangle of (x,y)=(3,4) returns a distance of 5
        """
        start = GeoPoint.from_rd(coords=(140000, 465000))
        end = GeoPoint.from_rd(coords=(140003, 465004))
        line = GeoPolyline(start, end)
        line = Polyline.from_geo_polyline(line)
        self.assertAlmostEqual(get_geo_polyline_section_length(line).sum(), 5., 7)

    def test__get_map_payload(self):
        """
        Check that bounding box, width and height are as expected.

        Draw a line with intervals of 0.5 metres, from the bottom left to the top right, where three sections are
        expected
        :return:
        """
        d_distance = 0.25 * np.sqrt(2)
        start = GeoPoint.from_rd(coords=(140000 - d_distance, 465000 - d_distance))
        end = GeoPoint.from_rd(coords=(140000 + d_distance, 465000 + d_distance))
        payload_dict = _get_map_payload(points_list=[start.rd, end.rd], interval=0.5)
        self.assertEqual(payload_dict['width'], '3')
        self.assertEqual(payload_dict['height'], '3')
        for verification_value, calculated_value in zip(
                [140000 - 1.5 * d_distance, 465000 - 1.5 * d_distance, 140000 + 1.5 * d_distance,
                 465000 + 1.5 * d_distance],
                [float(s) for s in payload_dict['bbox'].split(',')]):
            self.assertAlmostEqual(calculated_value, verification_value, 4)

    def test_ahn_value_at_point(self):
        """
        Check that the AHN value at (140000, 465000) == 1.1080 at ahn3_05m_dtm
        """
        d_distance = 0.25 * np.sqrt(2)
        start = GeoPoint.from_rd(coords=(140000 - d_distance, 465000 - d_distance))
        end = GeoPoint.from_rd(coords=(140000 + d_distance, 465000 + d_distance))
        line = GeoPolyline(start, end)
        line = Polyline.from_geo_polyline(line)
        ahn_value = get_ahn_of_polyline(polyline=line, interval=0.5)[0]['ahn_values'][1]
        self.assertAlmostEqual(ahn_value, 1.1080, 7)

    def test_ahn_value_at_point_with_horizontal_line(self):
        """
        Check that the AHN value at (140000, 465000) == 1.1080 at ahn3_05m_dtm
        """
        d_distance = 0.5
        start = GeoPoint.from_rd(coords=(140000 - d_distance, 465000))
        end = GeoPoint.from_rd(coords=(140000 + d_distance, 465000))
        line = GeoPolyline(start, end)
        line = Polyline.from_geo_polyline(line)
        ahn_value = get_ahn_of_polyline(polyline=line, interval=0.5)[0]['ahn_values'][1]
        self.assertAlmostEqual(ahn_value, 1.1080, 7)

    def test_ahn_value_at_point_with_vertical_line(self):
        """
        Check that the AHN value at (140000, 465000) == 1.1080 at ahn3_05m_dtm
        """
        d_distance = 0.5
        start = GeoPoint.from_rd(coords=(140000, 465000 - d_distance))
        end = GeoPoint.from_rd(coords=(140000, 465000 + d_distance))
        line = GeoPolyline(start, end)
        line = Polyline.from_geo_polyline(line)
        ahn_value = get_ahn_of_polyline(polyline=line, interval=0.5)[0]['ahn_values'][1]
        self.assertAlmostEqual(ahn_value, 1.1080, 7)

    def test_ahn_value_at_point_for_linked_lines(self):
        """
        Check that a polyline with two linked lines ends the first line with the same value as the second one ends with
        """
        d_distance = 0.5
        start_horizontal_left = GeoPoint.from_rd(coords=(140000 - d_distance, 465000))
        end_horizontal_left = GeoPoint.from_rd(coords=(140000, 465000))
        end_vertical_up = GeoPoint.from_rd(coords=(140000, 465000 + d_distance))
        line = GeoPolyline(start_horizontal_left, end_horizontal_left, end_vertical_up)
        line = Polyline.from_geo_polyline(line)
        ahn_dict_list = get_ahn_of_polyline(polyline=line, interval=0.5)
        self.assertAlmostEqual(ahn_dict_list[0]['ahn_values'][-1], ahn_dict_list[1]['ahn_values'][0], 7)
