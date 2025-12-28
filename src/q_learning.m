function [q_table, training_history] = q_learning(num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon, maze_size, obstacle_density)
% Trains a Q-Learning agent using local state encoding with reward shaping.

    NUM_STATES = 3240;
    NUM_ACTIONS = 4;
    q_table = zeros(NUM_STATES, NUM_ACTIONS);
    
    rows = maze_size(1);
    cols = maze_size(2);
    base_density = obstacle_density;
    
    % Rewards
    GOAL_REWARD = 10;
    OBSTACLE_PENALTY = -5;
    STEP_PENALTY = -0.1;
    CLOSER_REWARD = 0.5;
    FARTHER_PENALTY = -0.3;
    
    % Tracking
    WINDOW_SIZE = 100;
    history = zeros(WINDOW_SIZE, 1);
    
    BAR_WIDTH = 30;
    UPDATE_FREQ = floor(num_episodes / 100);
    
    % Live plot
    show_plot = usejava('desktop');
    if show_plot
        fig = figure('Name', 'Training Progress', 'Position', [100, 100, 900, 600]);
    else
        fig = [];
    end
    
    fprintf('Training with Reward Shaping...\n');
    fprintf('Maze: %dx%d | Density: %.0f%% | Episodes: %d\n\n', rows, cols, base_density*100, num_episodes);
    
    % Baseline test
    NUM_TEST_MAZES = 50;
    [baseline_rate, baseline_steps] = run_eval(q_table, rows, cols, base_density, NUM_TEST_MAZES);
    
    log_episodes = [0];
    log_test_rate = [baseline_rate];
    log_avg_steps = [baseline_steps];
    log_epsilon = [epsilon];
    
    fprintf('[BASELINE] Episode 0 | Test: %.1f%% | AvgSteps: %.0f\n\n', baseline_rate, baseline_steps);
    
    for episode = 1:num_episodes
        obs_density = base_density + (rand() * 0.1 - 0.05);
        
        start_pos = [randi(rows), randi(cols)];
        goal_pos = [randi(rows), randi(cols)];
        min_dist = floor(rows * 0.6);
        while isequal(start_pos, goal_pos) || ...
              (abs(start_pos(1)-goal_pos(1)) + abs(start_pos(2)-goal_pos(2))) < min_dist
            goal_pos = [randi(rows), randi(cols)];
        end
        
        maze = create_maze(rows, cols, obs_density, goal_pos, start_pos);
        
        current_pos = start_pos;
        last_action_1 = 0;
        last_action_2 = 0;
        state = encode_state(maze, current_pos, goal_pos, last_action_1, last_action_2);
        
        is_done = false;
        step_count = 0;
        MAX_STEPS = rows * cols;
        
        while ~is_done && step_count < MAX_STEPS
            step_count = step_count + 1;
            old_dist = abs(current_pos(1) - goal_pos(1)) + abs(current_pos(2) - goal_pos(2));
            
            if rand() < epsilon
                action = randi(NUM_ACTIONS);
            else
                [~, action] = max(q_table(state, :));
            end
            
            next_pos = current_pos;
            switch action
                case 1, next_pos(1) = next_pos(1) - 1;
                case 2, next_pos(1) = next_pos(1) + 1;
                case 3, next_pos(2) = next_pos(2) - 1;
                case 4, next_pos(2) = next_pos(2) + 1;
            end
            
            hit_wall = false;
            if next_pos(1) < 1 || next_pos(1) > rows || ...
               next_pos(2) < 1 || next_pos(2) > cols || ...
               maze(next_pos(1), next_pos(2)) == 1
                hit_wall = true;
                next_pos = current_pos;
            end
            
            new_dist = abs(next_pos(1) - goal_pos(1)) + abs(next_pos(2) - goal_pos(2));
            
            if isequal(next_pos, goal_pos)
                reward = GOAL_REWARD;
                is_done = true;
                history(mod(episode-1, WINDOW_SIZE)+1) = 1;
            elseif hit_wall
                reward = OBSTACLE_PENALTY;
            else
                reward = STEP_PENALTY;
                if new_dist < old_dist
                    reward = reward + CLOSER_REWARD;
                elseif new_dist > old_dist
                    reward = reward + FARTHER_PENALTY;
                end
            end
            
            next_state = encode_state(maze, next_pos, goal_pos, action, last_action_1);
            
            old_q = q_table(state, action);
            max_next_q = max(q_table(next_state, :));
            q_table(state, action) = old_q + alpha * (reward + gamma * max_next_q - old_q);
            
            state = next_state;
            current_pos = next_pos;
            last_action_2 = last_action_1;
            last_action_1 = action;
        end
        
        if ~is_done
            history(mod(episode-1, WINDOW_SIZE)+1) = 0;
        end
        
        if epsilon > min_epsilon
            epsilon = epsilon * epsilon_decay;
        end
        
        % Progress update
        if mod(episode, UPDATE_FREQ) == 0 || episode == num_episodes
            progress = episode / num_episodes;
            filled = round(progress * BAR_WIDTH);
            bar_str = [repmat('█', 1, filled), repmat('░', 1, BAR_WIDTH - filled)];
            
            [test_rate, avg_steps] = run_eval(q_table, rows, cols, base_density, NUM_TEST_MAZES);
            
            log_episodes = [log_episodes, episode];
            log_test_rate = [log_test_rate, test_rate];
            log_avg_steps = [log_avg_steps, avg_steps];
            log_epsilon = [log_epsilon, epsilon];
            
            if ishandle(fig)
                update_training_plot(fig, log_episodes, log_test_rate, log_avg_steps, log_epsilon, episode, num_episodes);
            end
            
            fprintf('\r[%s] %5.1f%% | Ep: %d/%d | Test: %5.1f%% | AvgSteps: %.0f | ε: %.3f', ...
                bar_str, progress*100, episode, num_episodes, test_rate, avg_steps, epsilon);
        end
    end
    
    % Final evaluation
    fprintf('\n\nFinal Evaluation on 100 unseen mazes...\n');
    [final_rate, final_avg_steps] = run_eval(q_table, rows, cols, base_density, 100);
    fprintf('Training Complete!\n');
    fprintf('  Final Test Success Rate: %.1f%%\n', final_rate);
    fprintf('  Final Avg Steps: %.1f\n', final_avg_steps);
    
    training_history.episodes = log_episodes;
    training_history.test_rate = log_test_rate;
    training_history.avg_steps = log_avg_steps;
    training_history.epsilon = log_epsilon;
    training_history.final_test_rate = final_rate;
    training_history.final_avg_steps = final_avg_steps;
end


function [success_rate, avg_steps] = run_eval(q_table, rows, cols, obstacle_density, num_tests)
% Evaluates agent on unseen mazes using greedy policy.

    successes = 0;
    total_steps = 0;
    successful_tests = 0;
    
    for t = 1:num_tests
        [success, steps] = test_on_unseen_maze(q_table, rows, cols, obstacle_density);
        successes = successes + success;
        if success
            total_steps = total_steps + steps;
            successful_tests = successful_tests + 1;
        end
    end
    
    success_rate = (successes / num_tests) * 100;
    if successful_tests > 0
        avg_steps = total_steps / successful_tests;
    else
        avg_steps = rows * cols * 2;
    end
end


function [success, steps] = test_on_unseen_maze(q_table, rows, cols, obstacle_density)
% Tests agent on a single unseen maze with greedy policy.

    start_pos = [randi(rows), randi(cols)];
    goal_pos = [randi(rows), randi(cols)];
    min_dist = floor(rows * 0.6);
    while isequal(start_pos, goal_pos) || ...
          (abs(start_pos(1)-goal_pos(1)) + abs(start_pos(2)-goal_pos(2))) < min_dist
        goal_pos = [randi(rows), randi(cols)];
    end
    
    maze = create_maze(rows, cols, obstacle_density, goal_pos, start_pos);
    
    current_pos = start_pos;
    last_action_1 = 0;
    last_action_2 = 0;
    MAX_STEPS = rows * cols / 2;
    
    for step = 1:MAX_STEPS
        state = encode_state(maze, current_pos, goal_pos, last_action_1, last_action_2);
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
           maze(next_pos(1), next_pos(2)) == 1
            next_pos = current_pos;
        end
        
        if isequal(next_pos, goal_pos)
            success = 1;
            steps = step;
            return;
        end
        
        last_action_2 = last_action_1;
        last_action_1 = action;
        current_pos = next_pos;
    end
    
    success = 0;
    steps = MAX_STEPS;
end


function update_training_plot(fig, episodes, test_rate, avg_steps, epsilon_log, current_ep, total_ep)
% Updates the live training visualization.

    subplot(2, 2, 1);
    plot(episodes, test_rate, 'b-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('Success Rate (%)');
    title('Generalization Test Rate');
    ylim([0, 100]); grid on;
    
    subplot(2, 2, 2);
    plot(episodes, avg_steps, 'r-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('Avg Steps');
    title('Average Steps to Goal');
    grid on;
    
    subplot(2, 2, 3);
    plot(episodes, epsilon_log, 'g-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('Epsilon');
    title('Exploration Rate (ε)');
    ylim([0, 1]); grid on;
    
    subplot(2, 2, 4);
    cla; axis off;
    text(0.1, 0.8, sprintf('Episode: %d / %d', current_ep, total_ep), 'FontSize', 12);
    text(0.1, 0.6, sprintf('Test Rate: %.1f%%', test_rate(end)), 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'b');
    text(0.1, 0.4, sprintf('Avg Steps: %.1f', avg_steps(end)), 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
    text(0.1, 0.2, sprintf('Epsilon: %.3f', epsilon_log(end)), 'FontSize', 12, 'Color', [0 0.5 0]);
    
    drawnow;
end
