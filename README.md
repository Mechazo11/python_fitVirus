# python_fitVirus
Port of Dr. Batista's single wave fitVirus MATLAB code 
Reference -- M. Batista (2020). fitVirus (https://www.mathworks.com/matlabcentral/fileexchange/74411-fitvirus), MATLAB Central File Exchange. Retrieved November 2, 2020.

//TO-Do insert image of plot generated

## Important Notes ---------------------------------------------------------------------------

<ol>
  <li>Input must be a cumulative case data. Sample dataset is in 'day' level but will work for state level and country level dataset</li>
  <li>All values from iniGuess(C) must be positive values</li>
  <li>x-axis data shows day as integer array. New code has to be added to convert it to proper date format i.e. mm/dd/yy</li>
  <li>As long as R2 score is above, 0.97, the model is considered to be fit. This is an emperical observation which needs further investigation</li>
</ol>

## Usage Notes ---------------------------------------------------------------------------------

<ol>
  <li>"funcs.py" MUST be in the same folder as the "python_fitvirus.ipynb" notebook</li>
  <li>Requires installation of GEKKO package, installation instruction can be found here: https://github.com/BYU-PRISM/GEKKO</li>
  <li>passing [plot_on = False] will supress plot generation but it has to be done for each of the three type of plots</li>
  <li>master_plot boolean variable controls whether we can see the final plot with emperical phases or not</li>
  <li>master_plot's y axis is min-maxed normalized but has some sections HARDCODED which needs to be changed if you want to plot phase rectangles on either cumulative or rate of change dataset</li>
</ol>
