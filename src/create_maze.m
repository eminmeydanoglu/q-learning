function maze = create_maze(rows, cols, obstacle_density, goal_pos, start_pos)
% Creates a random solvable maze with guaranteed path from start to goal.

    MAX_ATTEMPTS = 50;
    
    for attempt = 1:MAX_ATTEMPTS
        maze = zeros(rows, cols);
        num_obstacles = floor(rows * cols * obstacle_density);
        obstacles_placed = 0;
        failed_placements = 0;
        max_failed = num_obstacles;
        CHECK_INTERVAL = 5;
        
        while obstacles_placed < num_obstacles && failed_placements < max_failed
            r = randi(rows);
            c = randi(cols);
            
            if maze(r, c) ~= 0 || isequal([r, c], goal_pos) || isequal([r, c], start_pos)
                continue;
            end
            
            maze(r, c) = 1;
            obstacles_placed = obstacles_placed + 1;
            
            if mod(obstacles_placed, CHECK_INTERVAL) == 0
                if ~check_path_exists_internal(maze, start_pos, goal_pos)
                    maze(r, c) = 0;
                    obstacles_placed = obstacles_placed - 1;
                    failed_placements = failed_placements + 1;
                end
            end
        end

        maze(goal_pos(1), goal_pos(2)) = 2;
        maze(start_pos(1), start_pos(2)) = 3;
        
        if check_path_exists_internal(maze, start_pos, goal_pos)
            return;
        end
    end
    
    warning('Could not place all obstacles while maintaining path. Placed %d/%d obstacles.', ...
        obstacles_placed, num_obstacles);
end


function path_exists = check_path_exists_internal(maze, start_pos, goal_pos)
% BFS to verify path exists from start to goal.

    [rows, cols] = size(maze);
    visited = false(rows, cols);
    queue = start_pos;
    visited(start_pos(1), start_pos(2)) = true;
    directions = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    while ~isempty(queue)
        current_pos = queue(1, :);
        queue(1, :) = [];
        
        if isequal(current_pos, goal_pos)
            path_exists = true;
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
            queue = [queue; next_pos];
        end
    end
    
    path_exists = false;
end
