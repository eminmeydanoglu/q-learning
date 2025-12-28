% main.m - Q-Learning Pathfinding Agent
% Usage: main (train new) | agent_file='path.mat'; main (load existing)

addpath(genpath('.'));

AGENT_FILE = '';
if exist('agent_file', 'var')
    AGENT_FILE = agent_file;
end

% Parameters
MAZE_SIZE = [10, 10];
OBSTACLE_DENSITY = 0.20;
NUM_EPISODES = 12000;
ALPHA = 0.1;
GAMMA = 0.99;
EPSILON = 1.0;
EPSILON_DECAY = 0.9995;
MIN_EPSILON = 0.05;
NUM_TESTS = 24;

%% Train or Load Agent

if ~isempty(AGENT_FILE) && exist(AGENT_FILE, 'file')
    fprintf('\n========================================\n');
    fprintf('       LOADING PRE-TRAINED AGENT       \n');
    fprintf('========================================\n');
    fprintf('File: %s\n', AGENT_FILE);
    
    loaded = load(AGENT_FILE);
    agent_data = loaded.agent_data;
    q_table = agent_data.q_table;
    MAZE_SIZE = agent_data.maze_config.size;
    OBSTACLE_DENSITY = agent_data.maze_config.obstacle_density;
    
    fprintf('Maze: %dx%d | Density: %.0f%%\n', MAZE_SIZE(1), MAZE_SIZE(2), OBSTACLE_DENSITY * 100);
    fprintf('Trained at: %s\n', agent_data.trained_at);
    fprintf('========================================\n\n');
else
    fprintf('\n========================================\n');
    fprintf('       TRAINING Q-LEARNING AGENT       \n');
    fprintf('========================================\n\n');
    
    t_start = tic;
    [q_table, training_history] = q_learning(NUM_EPISODES, ALPHA, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON, MAZE_SIZE, OBSTACLE_DENSITY);
    duration = toc(t_start);
    
    fprintf('Training took %.1f seconds.\n', duration);
    
    % Save agent
    script_dir = fileparts(mfilename('fullpath'));
    agents_dir = fullfile(script_dir, 'agents');
    if ~exist(agents_dir, 'dir')
        mkdir(agents_dir);
    end
    
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = fullfile(agents_dir, sprintf('agent_%s_%dx%d.mat', timestamp, MAZE_SIZE(1), MAZE_SIZE(2)));
    
    agent_data.q_table = q_table;
    agent_data.training_history = training_history;
    agent_data.parameters.alpha = ALPHA;
    agent_data.parameters.gamma = GAMMA;
    agent_data.parameters.epsilon = EPSILON;
    agent_data.parameters.epsilon_decay = EPSILON_DECAY;
    agent_data.parameters.min_epsilon = MIN_EPSILON;
    agent_data.parameters.num_episodes = NUM_EPISODES;
    agent_data.maze_config.size = MAZE_SIZE;
    agent_data.maze_config.obstacle_density = OBSTACLE_DENSITY;
    agent_data.trained_at = timestamp;
    agent_data.training_duration = duration;
    
    save(filename, 'agent_data');
    fprintf('✓ Agent saved to: %s\n', filename);
end


%% Test Agent

fprintf('\n========================================\n');
fprintf('   TESTING ON %d RANDOM MAZES   \n', NUM_TESTS);
fprintf('========================================\n\n');

test_results = zeros(NUM_TESTS, 1);
rows = MAZE_SIZE(1); 
cols = MAZE_SIZE(2);

num_cols = ceil(sqrt(NUM_TESTS * 4/3));
num_rows = ceil(NUM_TESTS / num_cols);

figure('Name', 'Agent Test Results', 'Position', [50, 50, 1400, 700]);

for t = 1:NUM_TESTS
    start_pos = [randi(rows), randi(cols)];
    goal_pos = [randi(rows), randi(cols)];
    while isequal(start_pos, goal_pos)
        goal_pos = [randi(rows), randi(cols)];
    end
    
    test_maze = create_maze(rows, cols, OBSTACLE_DENSITY, goal_pos, start_pos);
    
    current_pos = start_pos;
    last_action_1 = 0;
    last_action_2 = 0;
    path = current_pos;
    max_steps = rows * cols * 2;
    is_solved = false;
    
    for step = 1:max_steps
        state = encode_state(test_maze, current_pos, goal_pos, last_action_1, last_action_2);
        [~, action] = max(q_table(state, :));
        
        next_pos = current_pos;
        switch action
            case 1, next_pos(1) = next_pos(1) - 1;
            case 2, next_pos(1) = next_pos(1) + 1;
            case 3, next_pos(2) = next_pos(2) - 1;
            case 4, next_pos(2) = next_pos(2) + 1;
        end
        
        if next_pos(1) < 1 || next_pos(1) > rows || ...
           next_pos(2) < 1 || next_pos(2) > cols || ...
           test_maze(next_pos(1), next_pos(2)) == 1
            next_pos = current_pos;
        end
        
        path = [path; next_pos];
        
        if isequal(next_pos, goal_pos)
            is_solved = true;
            break;
        end
        
        last_action_2 = last_action_1;
        last_action_1 = action;
        current_pos = next_pos;
    end
    
    if is_solved
        test_results(t) = 1;
        status = '✓ SUCCESS';
        status_color = 'green';
    else
        test_results(t) = 0;
        status = '✗ FAIL';
        status_color = 'red';
    end
    
    % Visualize
    subplot(num_rows, num_cols, t);
    img = ones(rows, cols, 3);
    for r = 1:rows
        for c = 1:cols
            if test_maze(r, c) == 1
                img(r, c, :) = [0.2, 0.2, 0.2];
            end
        end
    end
    
    imagesc(img);
    hold on;
    if ~isempty(path)
        plot(path(:, 2), path(:, 1), 'r-', 'LineWidth', 1.5);
    end
    plot(start_pos(2), start_pos(1), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot(goal_pos(2), goal_pos(1), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    title(sprintf('Test %d: %s', t, status), 'Color', status_color);
    axis equal tight;
    set(gca, 'XTick', [], 'YTick', []);
    hold off;
    
    fprintf('Test %d: %s\n', t, status);
    drawnow;
end

% Summary
success_count = sum(test_results);
success_rate = success_count / NUM_TESTS * 100;

fprintf('\n========================================\n');
fprintf('            TEST SUMMARY               \n');
fprintf('========================================\n');
fprintf('Total: %d | Success: %d (%.1f%%) | Fail: %d (%.1f%%)\n', ...
    NUM_TESTS, success_count, success_rate, NUM_TESTS - success_count, 100 - success_rate);
fprintf('========================================\n');
