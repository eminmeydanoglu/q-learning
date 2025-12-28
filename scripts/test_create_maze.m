% test_create_maze.m - Maze generation testing script

addpath(genpath('.'));

fprintf('\n========================================\n');
fprintf('       MAZE GENERATION TEST            \n');
fprintf('========================================\n\n');

test_cases = [
    10, 10, 0.1;
    10, 10, 0.2;
    10, 10, 0.3;
    20, 20, 0.2;
    50, 50, 0.2;
];

figure('Name', 'Maze Generation Tests', 'Position', [100, 100, 1400, 800]);

for i = 1:size(test_cases, 1)
    rows = test_cases(i, 1);
    cols = test_cases(i, 2);
    density = test_cases(i, 3);
    
    start_pos = [1, 1];
    goal_pos = [rows, cols];
    
    fprintf('Test %d: %dx%d maze with %.0f%% obstacles\n', i, rows, cols, density*100);
    
    maze = create_maze(rows, cols, density, goal_pos, start_pos);
    
    total_cells = rows * cols;
    obstacle_count = sum(maze(:) == 1);
    actual_density = obstacle_count / total_cells * 100;
    
    fprintf('  Obstacles: %d (%.1f%% actual vs %.1f%% requested)\n', ...
        obstacle_count, actual_density, density*100);
    
    [path_exists, path_length] = check_path_exists(maze, start_pos, goal_pos);
    
    if path_exists
        fprintf('  ✓ Path exists (length: %d)\n\n', path_length);
    else
        fprintf('  ✗ NO PATH!\n\n');
    end
    drawnow;
    
    subplot(2, 3, i);
    visualize_maze(maze);
    
    if path_exists
        title(sprintf('%dx%d (%.0f%%) - Path: %d', rows, cols, density*100, path_length), 'Color', 'green');
    else
        title(sprintf('%dx%d (%.0f%%) - NO PATH!', rows, cols, density*100), 'Color', 'red');
    end
end

%% Stress Test
fprintf('========================================\n');
fprintf('       PATH EXISTENCE STRESS TEST      \n');
fprintf('========================================\n\n');

num_trials = 100;
maze_size = 20;
densities = [0.1, 0.2, 0.3, 0.4];

fprintf('Running %d trials per density level...\n\n', num_trials);
drawnow;

subplot(2, 3, 6);
success_rates = zeros(size(densities));

for d = 1:length(densities)
    density = densities(d);
    successes = 0;
    
    for trial = 1:num_trials
        maze = create_maze(maze_size, maze_size, density, [maze_size, maze_size], [1, 1]);
        [path_exists, ~] = check_path_exists(maze, [1, 1], [maze_size, maze_size]);
        if path_exists
            successes = successes + 1;
        end
    end
    
    success_rates(d) = successes / num_trials * 100;
    fprintf('Density %.0f%%: %d/%d solvable (%.1f%%)\n', density*100, successes, num_trials, success_rates(d));
    drawnow;
end

bar(densities * 100, success_rates);
xlabel('Obstacle Density (%)');
ylabel('Solvable Mazes (%)');
title('Path Existence vs Obstacle Density');
ylim([0, 100]);
grid on;

fprintf('\n========================================\n');
fprintf('         TESTS COMPLETE!               \n');
fprintf('========================================\n');


%% Helper Functions

function visualize_maze(maze)
    [rows, cols] = size(maze);
    img = ones(rows, cols, 3);
    
    for r = 1:rows
        for c = 1:cols
            switch maze(r, c)
                case 1, img(r, c, :) = [0, 0, 0];
                case 2, img(r, c, :) = [0, 0.8, 0];
                case 3, img(r, c, :) = [0, 0.4, 0.8];
            end
        end
    end
    
    imagesc(img);
    axis equal tight;
    set(gca, 'XTick', [], 'YTick', []);
end


function [path_exists, path_length] = check_path_exists(maze, start_pos, goal_pos)
    [rows, cols] = size(maze);
    visited = false(rows, cols);
    queue = [start_pos, 0];
    visited(start_pos(1), start_pos(2)) = true;
    directions = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    while ~isempty(queue)
        current = queue(1, :);
        queue(1, :) = [];
        
        current_pos = current(1:2);
        current_dist = current(3);
        
        if isequal(current_pos, goal_pos)
            path_exists = true;
            path_length = current_dist;
            return;
        end
        
        for d = 1:4
            next_pos = current_pos + directions(d, :);
            
            if next_pos(1) < 1 || next_pos(1) > rows || ...
               next_pos(2) < 1 || next_pos(2) > cols
                continue;
            end
            
            if maze(next_pos(1), next_pos(2)) == 1 || visited(next_pos(1), next_pos(2))
                continue;
            end
            
            visited(next_pos(1), next_pos(2)) = true;
            queue = [queue; next_pos, current_dist + 1];
        end
    end
    
    path_exists = false;
    path_length = -1;
end
