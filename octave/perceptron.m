%% CS156 Learning From Data - Homework 1: The Perceptron Learning Algorithm
%
%  Instructions
%  ------------
% 
%  These file contains code that implement the perceptron
%  exercise. For additional info on octave https://www.octave.org. 
%
%     perceptron.m
%     plotData.m
%     train_data1/2.dat
%     train_params1/2.dat
%

%% Initialization
clear ; close all; clc

for j = 1:2
  %% Load Data
  text = sprintf(['train_data%d.dat'], j);
  data = load(text);
  text = sprintf(['train_params%d.dat'], j);
  params = load(text);

  %  The first two columns contains the exam scores and the third column
  %  contains the label.
  iterations = params(1,8);
  data_segment_size = params(1,5);
  m = data_segment_size;
  total_data = length(data);
  trials = total_data/m;
  N_cvdata = 1000;
  cvData = createData(N_cvdata);

  % Create New Figure
  TitleText = sprintf(['2-D Percepton, %d runs, %d pts/runs, %d max iterations'], trials, m, iterations);
  fax = figure("name",TitleText, "numbertitle",'off', "position", [200 100 1060 820]);
  set(gca, "linewidth", 8, "fontsize", 18);

  xlabel ("x");
  ylabel ("y");


  %% ==================== Part 1: Plotting ====================
  %  We start the exercise by first plotting the data to understand the 
  %  the problem we are working with.
  num_plots = 10;
  plotList1 = [1,2,3,4,5,6,7,8,9,10];
  plotList2 = [1,2,3,4,5,6,7,8,9,10];
  ##plotList2 = [1,20,30,40,50,60,70,80,90,100];
  hax = zeros(num_plots); # number of subplots
  for i = 1:num_plots
      if j=1
        I = plotList1(i);
      else
        I = plotList2(i);
      endif;
      startpt = m*(I-1)+1;
      endpt   = m*(I);
      cvX = cvData(1:N_cvdata, [1, 2]);
      X = data(startpt:endpt, [1, 2]); y = data(startpt:endpt, 3);
      if (params(I,5) != m)
        m = params(I,1);
        fprintf('Error with data segment... TODO-implement different data_segment_sizes');
        exit;
      endif;
      m1 = params(I,1); # Target
      b1 = params(I,2);
      m2 = params(I,3); # Learned
      b2 = params(I,4);
      c_score = params(I,6);
      g_score = params(I,7);
      fprintf(['Plotting new data randomly created between -1, +1 with + indicating (y = 1) examples and o ' ...
               'indicating (y = 0) examples. c=%f, g=%f\n'],c_score,g_score);
      ##pause;
      
##      T1 = cvX(:,1)*m1 + b1; ## X(:,1)*m1 + b1;
##      T2 = cvX(:,1)*m2 + b2; ## X(:,1)*m2 + b2;
##      Diff1 = cvX(:,2) - T1;
##      Diff2 = cvX(:,2) - T2;
      ##      Diff1 = X(:,2) - T1;
      ##      Diff2 = X(:,2) - T2;
##      cvY = (Diff1 > 0);
##      cvY2 = (Diff2 > 0);

      hax(i) = plotData(X, y, I, c_score, g_score);
##      if ((j == 1) & (i == 7))
##        ##hax(i) = fax;
##        hax(i) = onePlotData(cvX, cvY, I, cvpfxgx);
##      else
##        continue;
##        ##hax(i) = plotData(cvX, cvY, I, cvpfxgx);
##      endif;

      M1 = [m1;1]; # Target
      M2 = [m2;1]; # Learned
      fprintf(['Plotting lines matching the Target function\n' ...
              '   y = %fx + %f\n' ...
              'And the learned hypothesis:\n' ...
              '   y = %fx + %f\n'],m1,b1,m2,b2);
      t = -1.5:0.1:1.5;
      t1 = m1*t + b1;
      t2 = m2*t + b2;
      
      hold (hax(i), "on");
      plot(t, t1, 'b', "linewidth", 6, ...
           t, t2, 'm', "linewidth", 4)
      #text (0, b1, "Target y-intercept");
      refresh;
      hold (hax(i), "off");

      T1 = X(:,1)*m1 + b1;
      T2 = X(:,1)*m2 + b2;
      Diff1 = X(:,2) - T1;
      Diff2 = X(:,2) - T2;

      outliers = find((Diff1 > 0) & (Diff2 <0));
      hold (hax(i), "on"); 
      %  Draw a red circle around the outliers
##      plot(cvX(outliers,1), cvX(outliers,2), 'ro', 'LineWidth', 2, 'MarkerSize', 15);
      plot(X(outliers,1), X(outliers,2), 'ro', 'LineWidth', 2, 'MarkerSize', 15);
      refresh;
      hold (hax(i), "off");
  endfor;
  %% ==================== Part 2: ??? ====================
  %  That's all for now.
  %  We have two sets of data, one for N = 10, the other for N = 100

  
  endfor
