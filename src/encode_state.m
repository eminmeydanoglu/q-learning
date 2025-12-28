function state_idx = encode_state(maze, current_pos, goal_pos, last_action, ~)
% Encodes agent state using local vision + goal direction + action memory.
% State space: Vision(81) × Direction(8) × LastAction(5) = 3240 states

    [rows, cols] = size(maze);
    r = current_pos(1);
    c = current_pos(2);

    % Extended vision per direction: 0=open, 1=blocked, 2=wall ahead
    up_d1 = (r <= 1) || (maze(r-1, c) == 1);
    up_d2 = (r <= 2) || (maze(r-2, c) == 1);
    if up_d1, vis_up = 1; elseif up_d2, vis_up = 2; else, vis_up = 0; end
    
    down_d1 = (r >= rows) || (maze(r+1, c) == 1);
    down_d2 = (r >= rows-1) || (maze(r+2, c) == 1);
    if down_d1, vis_down = 1; elseif down_d2, vis_down = 2; else, vis_down = 0; end
    
    left_d1 = (c <= 1) || (maze(r, c-1) == 1);
    left_d2 = (c <= 2) || (maze(r, c-2) == 1);
    if left_d1, vis_left = 1; elseif left_d2, vis_left = 2; else, vis_left = 0; end
    
    right_d1 = (c >= cols) || (maze(r, c+1) == 1);
    right_d2 = (c >= cols-1) || (maze(r, c+2) == 1);
    if right_d1, vis_right = 1; elseif right_d2, vis_right = 2; else, vis_right = 0; end
    
    vis_idx = vis_up*27 + vis_down*9 + vis_left*3 + vis_right;

    % Goal direction (8 octants)
    dr = goal_pos(1) - r;
    dc = goal_pos(2) - c;
    angle = atan2(-dr, dc) * 180 / pi; 
    if angle < 0, angle = angle + 360; end
    dir_idx = floor(mod(angle + 22.5, 360) / 45);

    % Combine: vision(81) × direction(8) × action(5)
    state_idx = (vis_idx * 40) + (dir_idx * 5) + last_action + 1;
end
