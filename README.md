Not really alternating least squares, but it was the closest approximation.  It alternates minimising the RMSE of two imputed marginals against their known counterparts until both losses are below some threshold.  It uses a "seed" matrix to help guide it, since for this particular problem, we can make strong assumptions about the shape of the matrix based on other data we have available.

This was originally intended for a professional project which has since been scrapped as the company is now defunct.

## Usage

Supply the convergence algorithm with two marginals, which must each sum to the same value, and a matrix that has the rough shape of the matrix you'd like to impute.  Optionally, include threshold for convergence, maximum iterations (to prevent lengthy training time), and a Boolean value to determine whether or not to print the loss value and iteration number at each iteration.

## License

Copyright © 2019

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
