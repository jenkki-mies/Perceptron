function hax = plotData(X, y, data_id, c_score, g_score)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
subplot_id = data_id; ##roundb(data_id / 10) + 1
text = sprintf(['cscore=%d P(error)= %2.1f %%'], c_score, g_score*100);
hax = subplot (5, 2, subplot_id);
title(hax, text);
hold (hax, "on");
set(gca, "linewidth", 8, "fontsize", 18);

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

#    set(gca, "linewidth", 8, "fontsize", 18)
    % Find Indices of Positive and Negative Examples
    pos = find(y==1); neg = find(y == 0);
    % Plot Examples
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 10);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 10);

% =========================================================================
hold (hax, "off");
##hold off; ## don't hold off until the lines are plotted
end;