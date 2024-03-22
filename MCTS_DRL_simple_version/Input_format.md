# Input circuit board format

The input PCB board is a matrix whose elements has the following meanings:

(1) the area that the path can pass has value 0;

(2) the area of obstacles that the path can not pass has value 1, for example, the body of a chip;

(3) the net to be connected is represented by a value larger than 1, pins of the same net are represented by the same number. For example, pins of net 1 have a value of 2.

Note that now we can only deal with simple circuit boards with the following limits:

(1) circuit only contains one layer, that is we do not need to consider via

(2) the total number of nets is smaller than 10, maybe it also works if the circuit has more than 10 nets, but it is best not to exceed 10. If we only have circuits with more than 10 nets, then keep the one with the least nets, and we can try. 

(3) each net contains two pins. 

(4) current board size (input matrix size) is 30`x`30. 
