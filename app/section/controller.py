"""Copyright (c) 2022 VIKTOR B.V.

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
from io import StringIO

import numpy as np
from munch import Munch
from viktor.core import ViktorController, UserException
from viktor.views import MapView, MapResult, MapPolyline, SVGResult, SVGView
import matplotlib.pyplot as plt
from app.section.parametrization import SectionParametrization
from app.functions.ahn import get_ahn_of_polyline, Polyline


class SectionController(ViktorController):
    """Controller class which acts as interface for the Section entity type."""
    label = "Section"
    parametrization = SectionParametrization(width=30)

    @MapView('Map', duration_guess=1)
    def get_map_view(self, params: Munch, **kwargs: dict) -> MapResult:
        """Set the map view on which the line can be drawn for which the AHN values are wished to be known"""
        features = []
        if params.geo_polyline:
            features.append(MapPolyline.from_geo_polyline(params.geo_polyline))
        return MapResult(features)

    @SVGView("AHN along line", duration_guess=5)
    def get_svg_view(self, params: Munch, **kwargs: dict) -> SVGResult:
        """
        Make a view to show the AHN profile corresponding to the created line in the input
        """
        if not params.geo_polyline:
            raise UserException("No line has been created yet. Please define one on the map")
        fig, axes = plt.subplots(1, 1)  # Create a figure with an axes object to plot on
        axes.grid(visible=True, which='both', axis='both', linestyle=':')
        axes.set_ylabel('AHN3 [m]')
        axes.set_xlabel('Distance along line [m]')
        # Get AHN values along the line from the webserver
        lines_ahn_data_list = get_ahn_of_polyline(polyline=Polyline.from_geo_polyline(params.geo_polyline),
                                                  interval=params.interval)
        total_length = 0
        for line_dict in lines_ahn_data_list:
            length, ahn_list = line_dict['length'], line_dict['ahn_values']
            ahn_list_filtered = [value for value in ahn_list if abs(value) < 1e4]  # Filter out very large values
            axes.plot(total_length + np.linspace(0, length, len(ahn_list_filtered)),
                            ahn_list_filtered)  # Plot the returned AHN values
            total_length += length
        svg_data = StringIO()  # Set up a string that can be used to write a file to
        fig.savefig(svg_data, format='svg')  # Save the figure to the IO string
        plt.close(fig)
        return SVGResult(svg_data)
