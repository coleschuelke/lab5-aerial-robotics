# Source File Notes

Students need only to modify the indicated functions in `structurecomputer.cc` and
`ballonfinder.cc`.

## Explanation of the `main*.cc` files

There are four files that begin with `main` in this directory:

### `main.cc`
This is the file that gets compiled when one runs 'make' in the build directory.  
Copy to `main.cc` whichever of the other three files below is appropriate to your
purpose.  For example, the command

`cp main_full.cc main.cc`

overwrites `main.cc` with `main_full.cc`. 

## `main_pre-lecture.cc`
`main` function prior to lecture:  This is the bare-bones file before the lecture
in which Dr. Humphreys showed examples of solving least squares problems and 
setting up a call to the `computeStructure()` function.

## `main_post-lecture.cc`
`main` function after lecture:  This is the file as it should look after the lecture
in which Dr. Humphreys showed examples of solving least squares problems and 
setting up a call to the `computeStructure()` function.

## `main_full.cc`
`main` function with full capability: This file is the one you will use
to build the fully implemented `locateBalloons` application.  It draws in images, perfors structure
computation, optionally performs calibration, and displays the final result.
