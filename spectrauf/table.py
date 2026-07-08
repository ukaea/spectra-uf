"""
Class used to tabulate nuerical and string values into a table which
can be printed on the command line.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TableFormat:
    """DataClass which controls the print formatting of variables passed
    to the Table class.
    """

    integer: str = ""
    floating: str = ".6e"
    string: str = ""
    header_padding: int = 2
    column_speration: int = 2
    column_seperator: str = " "


class Table:
    """
    The Table class takes a 2d list of values and tablulates them into a string based
    table which can be printed on the command line as a light weight method of summarising
    data
    """

    def __init__(
        self,
        rows: list[list],
        headers: list[str] = None,
        table_format: TableFormat = None,
    ):
        """Table class accepts rows and headers, and can be used to produce a string
        of the data in tabulated form

        Parameters
        ----------
        rows : list[list]
            A 2D list of the data to tabulate
        headers : list[str], optional
            A list of headers for the tabluated values, by default None
        table_format : TableFormat, optional
            A data class storing the formatting requirements of each
            data type, such as int, float etc..., by default None
        """
        # store user data
        self._rows: list[list] = rows
        self._headers: list[str] = headers

        # store format
        self._format: TableFormat = (
            TableFormat() if isinstance(table_format, type(None)) else table_format
        )

        # check shape of rows and headers
        self._n_rows: int = len(rows)
        self._row_length: int = self._check_inputs()

        # store rows as strings
        self._rows_as_strings: list[str] = None

    @property
    def format(self):
        """returns the TableFormat instanced used to control string
        formatting of various variables

        Returns
        -------
        format : TableFormat
            Returns the TableFormat instance associated with the
            current Table instance
        """
        return self._format

    def _check_inputs(self):
        """
        Checks the shape of the rows provided for congruence

        returns
        -------
        row_length : int
            Returns the length of the longest row
        """
        self._row_length = len(self._rows[0])
        for row in self._rows[1:]:
            if len(row) != self._row_length:
                msg = "length of rows in table must be equal"
                raise ValueError(msg)

        if self._headers:
            if len(self._headers) != self._row_length:
                msg = "length of headers in table must equal row length"
                raise ValueError(msg)

        return self._row_length

    def _convert_rows_to_strings(self):
        """converts rows of values to strings

        Returns
        -------
        rows_output : list[str]
            Returns a list of the rows represented in string format
        """
        rows_output = [self._headers] if self._headers else []

        type_formats = self._type_formats()

        # convert raw values to strings
        for row in self._rows:
            table_row = []
            for value in row:
                value_as_string = f"{value:{type_formats[type(value)]}}"
                table_row.append(value_as_string)
            rows_output.append(table_row)

        return rows_output

    def _column_widths(self):
        """Calculates the required column widths based on the TableFormat
        string formatting values

        Returns
        -------
        column_width : int
            Returns the required column width
        """
        column_widths = [0 for _ in range(self._row_length)]

        for row in self._rows_as_strings:
            for j, col_value in enumerate(row):
                col_value_length = len(col_value)
                if col_value_length > column_widths[j]:
                    column_widths[j] = col_value_length

        if self._headers:
            for j, header_value in enumerate(self._headers):
                header_value_length = len(header_value)
                if header_value_length > column_widths[j]:
                    column_widths[j] = header_value_length

        return column_widths

    def _type_formats(self):
        """returns the formatting type compatible with
        f-string formatting

        Returns
        -------
        type_formats : dict[type, str]
            _description_
        """
        type_formats = {
            str: self._format.string,
            np.str_: self._format.string,
            int: self._format.integer,
            np.integer: self._format.integer,
            float: self._format.floating,
            np.floating: self._format.floating,
        }

        return type_formats

    @property
    def string(self):
        """returns the tabulated values in string format

        Returns
        -------
        tabluated_values : str
            Returns the tabluated values as a string
        """
        # add dummy headers if non provided by user
        if not self._headers:
            self._headers = [f"value_{i}" for i in range(self._row_length)]

        # convert all values and headers to strings applying formatting for each type
        self._rows_as_strings = self._convert_rows_to_strings()

        # calculate the required maximum colum width
        column_widths = self._column_widths()

        # set formatting types lookup
        type_formats = self._type_formats()

        # set up list to store output rows
        table_rows = []

        # add headers with column widths as list of strings
        table_row_header = [
            f"{value: >{column_widths[i] + self._format.header_padding}{type_formats[type(value)]}}"
            for i, value in enumerate(self._headers)
        ]
        table_row_underscore = [
            "-" * (i + self._format.header_padding) for i in column_widths
        ]
        table_rows.append(table_row_header)
        table_rows.append(table_row_underscore)

        # add all values with column widths as list of strings
        for row in self._rows:
            table_row_values = [
                f"{value: >{column_widths[i] + self._format.header_padding}{type_formats[type(value)]}}"
                for i, value in enumerate(row)
            ]
            table_rows.append(table_row_values)

        # add column seperators
        column_sperators = self._format.column_seperator * self._format.column_speration
        rows_as_strings = [column_sperators.join(row) for row in table_rows]

        # return table as string
        table: str = "\n".join(rows_as_strings)

        # return tabulated values
        return table

    def __str__(self):
        """Dunder method to allow direct printing via the in built
        print method.

        Returns
        -------
        string : str
            returns the string representation of the tabulated values
        """
        return self.string
