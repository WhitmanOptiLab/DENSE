#include "model.hpp"
#include "color.hpp"
#include "csvr.hpp"

#include <cmath>



model::model(const string& pcfGradientFile, const string& pcfPerturbFile,
        const int& pcfTotalWidth) :
    _using_perturb(pcfPerturbFile.size() > 0),
    _using_gradients(pcfGradientFile.size() > 0)
{
    csvr *perturbFile = 0, *gradientFile = 0;
    
    RATETYPE global_pert_val = strtold(pcfPerturbFile.c_str(), 0);
    bool global_pert = (global_pert_val != 0.0);

    if (_using_perturb)
    {
        // Try loading file, suppress warning if string can be read as RATETYPE
        perturbFile = new csvr(pcfPerturbFile, global_pert);
        _using_perturb = perturbFile->is_open() || global_pert;
    }

    if (_using_gradients)
    {
        gradientFile = new csvr(pcfGradientFile);
        _using_gradients = gradientFile->is_open();
    }


    // perturbation factor, gradient low bound, gradient high bound,
    //   gradient width index start, gradient width index end, gradient slope
    RATETYPE tPert = 0.0, tGradY1 = 0.0, tGradY2 = 0.0,
             tGradX1 = 0.0, tGradX2 = 0.0, tGradM = 0.0;
    for (int i = 0; i < NUM_REACTIONS; i++)
    {
        // Perturb default (0.0 if argument was not a RATETYPE)
        // Prevents crashes in case perturbation parsing fails
        factors_perturb[i] = global_pert_val;

        if (_using_perturb && !global_pert)
        {
            if (perturbFile->get_next(&tPert))
            {
                factors_perturb[i] = tPert;
                tPert = 0.0;
            }
            else
            {
                // Error: Invalid number of filled cells
                cout << color::set(color::RED) <<
                    "CSV perturbations parsing failed. Ran out of cells to read "
                    "upon reaching reaction \"" << reaction_str[i] << "\"." <<
                    color::clear() << endl;
            }
        }

        
        // Gradient defaults
        // Prevents crashes in case gradient parsing fails
        _has_gradient[i] = false;
        factors_gradient[i] = NULL;

        if (_using_gradients)
        {
            // Read all tGrad--s
            if (gradientFile->get_next(&tGradX1) &&
                gradientFile->get_next(&tGradY1) &&
                gradientFile->get_next(&tGradX2) &&
                gradientFile->get_next(&tGradY2) )
            {
                if (tGradX1>=0 && tGradX2<=pcfTotalWidth)
                {
                    // If equal, more than likely, user does not want to
                    //   enable gradients for this specie
                    if (tGradX1!=tGradX2)
                    {
                        _has_gradient[i] = true;
                        factors_gradient[i] = new RATETYPE[pcfTotalWidth];
                        tGradM = (tGradY2 - tGradY1) / (tGradX2 - tGradX1);

                        for (int j=round(tGradX1); j<=round(tGradX2); j++)
                        {
                            factors_gradient[i][j] = tGradY1;
                            tGradY1 += tGradM;
                        }
                    }
                }
                else
                {
                    // Error: Invalid numbers in cells
                    cout << color::set(color::RED) <<
                        "CSV gradients parsing failed. Invalid grad_x1 and/or "
                        "grad_x2 setting(s) for reaction \"" << reaction_str[i] <<
                        "\"." << color::clear() << endl;
                }
            }
            else
            {
                // Error: Invalid number of filled cells
                cout << color::set(color::RED) <<
                    "CSV gradients parsing failed. Ran out of cells to read upon "
                    "reaching reaction \"" << reaction_str[i] << "\"." <<
                    color::clear() << endl;
            }
        }
    }


    if (perturbFile)
    {
        delete perturbFile;
        perturbFile = 0;
    }

    if (gradientFile)
    {
        delete gradientFile;
        gradientFile = 0;
    }
}
