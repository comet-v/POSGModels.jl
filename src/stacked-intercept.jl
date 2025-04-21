module StackedIntercept

using MarkovGames
using StaticArrays
using POMDPTools
using LinearAlgebra
using RecipesBase

export Coord, StackedInterceptMG, StackedInterceptState

const ACTION_DIRS = SA[
    SA[0, 1], # up
    SA[1, 0], # right
    SA[0,-1], # down
    SA[-1,0]  # left
]

const Coord = SVector{2, Int}

struct StackedInterceptState
    attackers::SVector{2,Coord} # could parameterize, but keep as 2 for now
    defender::Coord
    terminal::Bool
end

Base.@kwdef struct StackedInterceptMG{RM} <: MG{StackedInterceptState, Tuple{SVector{2,Int},Int}}
    reward_model    ::  RM              = DenseReward()
    tag_reward      ::  Float64         = 10.0
    sam_penalty     ::  Float64         = -5.0
    step_cost       ::  Float64         = 1.0
    discount        ::  Float64         = 0.95
    floor           ::  Coord           = Coord(11, 7)
    obstacles       ::  Set{Coord}      = Set{Coord}()
    SAMsites        ::  Set{Coord}      = Set{Coord}()
    initialstate    ::  StackedInterceptState  = StackedInterceptState(SA[Coord(1,1),Coord(2,1)], Coord(1,5), false)
    goal            ::  Set{Coord}      = Set{Coord}([Coord(11, 7)])
end

Base.@kwdef struct DenseReward
    scale   ::  Float64 = 1.0
    peak    ::  Float64 = 10.0
    d_max   ::  Float64 = 5.0
end

function (dr::DenseReward)(s1::Coord, s2::Coord)
    (;scale, peak, d_max) = dr
    d = norm(s2 .- s1, 2)
    return if iszero(d)
        peak
    elseif d > d_max
        0.0
    else
        (scale/d)
    end
end

struct SparseReward end

(sr::SparseReward)(s1::Coord, s2::Coord) = all(s1 .== s2)

function MarkovGames.states(p::StackedInterceptMG)
    v = map(CartesianIndices((p.floor[1], p.floor[2], p.floor[1], p.floor[2], p.floor[1], p.floor[2]))) do c_i
        t = Tuple(c_i)
        return StackedInterceptState(
            SA[Coord(t[1], t[2]), Coord(t[3], t[4])],
            Coord(t[5], t[6]), 
            false
        )
    end
    return push!(vec(v), StackedInterceptState(SA[Coord(1,1),Coord(1,1)], Coord(1,1), true)) 
end

MarkovGames.actions(p::StackedInterceptMG) = (CartesianIndices((1:4, 1:4)), CartesianIndices(1:4))

MarkovGames.discount(p::StackedInterceptMG) = p.discount

MarkovGames.initialstate(p::StackedInterceptMG) = Deterministic(p.initialstate)

function add_if_clear(floor, obstacles, s::Coord, a::Coord)
    sp = s .+ a
    if (any(sp .< Coord(1,1)) || any(sp .> floor) || sp ∈ obstacles)
        return s
    else
        return sp
    end
end

add_if_clear(floor::Coord, obstacles, s::Coord, a::Int) = add_if_clear(floor, obstacles, s, ACTION_DIRS[a])

function MarkovGames.transition(p::StackedInterceptMG{DenseReward}, s, (a1, a2))
    (;floor, obstacles) = p
    return if any(s_i ∈ p.goal for s_i in s.attackers)
        Deterministic(StackedInterceptState(SA[Coord(1,1),Coord(1,1)], Coord(1,1), true))
    else
        next_attackers = map(s.attackers, Tuple(a1)) do s_i, a_i
            add_if_clear(floor, obstacles, s_i, a_i)
        end
        next_defender = add_if_clear(floor, obstacles, s.defender, only(Tuple(a2)))
        return Deterministic(StackedInterceptState(next_attackers, next_defender, false))
    end
end

# function MarkovGames.transition(p::InterceptMG, s, (a1, a2))
#     (;floor, obstacles) = p
#     return if s.attacker == s.defender || s.attacker ∈ p.goal 
#         Deterministic(InterceptState(Coord(1,1), Coord(1,1), true))
#     else
#         next_attacker = add_if_clear(floor, obstacles, s.attacker, a1)
#         next_defender = add_if_clear(floor, obstacles, s.defender, a2)
#         Deterministic(InterceptState(next_attacker, next_defender, false))
#     end
# end

MarkovGames.isterminal(::StackedInterceptMG, s) = s.terminal

function MarkovGames.reward(p::StackedInterceptMG, s::StackedInterceptState, a)
    if isterminal(p, s)
        return 0.0 # No reward if already in terminal state
    end

    # Calculate next positions to determine rewards based on outcome
    a1, a2 = a
    (;floor, obstacles, goal, SAMsites, reward_model, tag_reward, sam_penalty, step_cost) = p

    next_attackers = map(s.attackers, Tuple(a1)) do s_i, a_i
        add_if_clear(floor, obstacles, s_i, a_i)
    end
    next_defender = add_if_clear(floor, obstacles, s.defender, only(Tuple(a2)))

    # --- Calculate Reward Components based on the *outcome* of action 'a' ---

    # 1. Goal Reward: Positive reward for each attacker reaching the goal
    goal_reward = 0.0
    for attacker_sp in next_attackers
        if attacker_sp ∈ goal
            goal_reward += tag_reward
        end
    end

    # 2. SAM Site Penalty: Negative reward for each attacker landing on a SAM site
    #    (unless it's also a goal square)
    sam_penalty_total = 0.0
    for attacker_sp in next_attackers
        if attacker_sp ∈ SAMsites && attacker_sp ∉ goal
            sam_penalty_total += sam_penalty # sam_penalty is negative
        end
    end

    # 3. Capture Penalty / Proximity Reward:
    #    Use reward_model for proximity. If captured, this gives a large value.
    #    We want negative reward for proximity from attacker's perspective.
    #    If captured, this should be a large negative reward.
    proximity_reward_total = 0.0
    capture_penalty = 0.0
    for attacker_sp in next_attackers
        if attacker_sp == next_defender
            # Apply a large penalty for being captured, potentially overriding proximity
            capture_penalty -= tag_reward # Use tag_reward as the magnitude of capture penalty
        else
            # Regular proximity reward (negative because attacker wants distance)
            proximity_reward_total -= reward_model(attacker_sp, next_defender)
        end
    end

    # 4. Step Cost: Small penalty for taking a step
    step_cost_total = -step_cost

    # Combine all reward components
    # Note: If goal is reached or capture occurs, the game transitions to terminal *next* step.
    # This reward is for the step *leading* to that terminal state.
    total_reward = goal_reward + sam_penalty_total + proximity_reward_total + capture_penalty + step_cost_total

    return total_reward
end

function stateindex(f::Coord, s::StackedInterceptState)
    (;attackers, defender, terminal) = s
    nc, nr = f
    if terminal
        n_agents = length(attackers) + 1
        n_pos = (nr * nc) ^ n_agents
        return n_pos+1
    else
        return LinearIndices((nc,nr,nc,nr,nc,nr))[attackers[1]..., attackers[2]..., defender...]
    end
end

MarkovGames.stateindex(p::StackedInterceptMG, s::StackedInterceptState) = stateindex(p.floor, s)

function MarkovGames.convert_s(::Type{Vector{T}} , s::StackedInterceptState, p::StackedInterceptMG) where T
    (; floor) = p
    attackers = map(s.attackers) do s_i
        (s_i .- floor ./ 2) ./ floor
    end
    defender = (s.defender .- floor ./ 2) ./ floor
    return T[
        attackers[1]..., 
        attackers[2]...,
        defender...
    ]
end

function MarkovGames.convert_s(::Type{StackedInterceptState}, x::AbstractVector, p::StackedInterceptMG)
    (; floor) = p
    # should get this from game type inference
    n_attackers = 2 # (length(x) ÷ 2) - 2
    attackers = map(StaticArrays.SUnitRange(1,n_attackers)) do idx
        idx0 = 2*(idx - 1) + 1
        idx1 = idx0 + 2
        attacker = x[idx0:idx1]
        return Coord(round.(Int, (attacker .* floor) .+ floor ./ 2))
    end
    defender = round.(Int, (x[3:4] .* floor) .+ floor ./ 2)
    return InterceptState(attackers, Coord(defender[1], defender[2]), false)
end

## visualization

function action_lines(x::Coord)
    return map(ACTION_DIRS) do a
        sp = x + a
        [x[1], sp[1]], [x[2], sp[2]]
    end
end

@recipe function f(game::StackedInterceptMG, s::StackedInterceptState)
    (;attackers, defender) = s
    xlims --> (0, game.floor[1]+1)
    ylims --> (0, game.floor[2]+1)
    xticks --> nothing
    yticks --> nothing
    goals = collect(game.goal)
    obstacles_vis = collect(game.obstacles) # Collect obstacles for plotting
    samsites_vis = collect(game.SAMsites) # Collect SAM sites for plotting

    @series begin
        seriestype  := :scatter
        c           --> [1,2,:red]
        [attackers[1][1], attackers[2][1], defender[1]], [attackers[1][2], attackers[2][2], defender[2]]
    end
    @series begin
        seriestype := :scatter
        ms := 20
        c := :yellow
        first.(goals), last.(goals)
    end

    if !isempty(obstacles_vis)
        @series begin
            seriestype := :scatter
            markershape := :square 
            markersize := 8      
            c := :grey           
            markerstrokewidth := 0
            first.(obstacles_vis), last.(obstacles_vis)
        end
    end

    # Plot SAM Sites
    if !isempty(samsites_vis)
        @series begin
            seriestype := :scatter
            markershape := :square
            markersize := 15 # Adjust size to fill cell
            markerstrokewidth := 0
            c := :purple
            # Filter out SAM sites that are also goals or obstacles to avoid overplotting
            samsites_plot = filter(site -> site ∉ goals && site ∉ obstacles_vis, samsites_vis)
            first.(samsites_plot), last.(samsites_plot)
        end
    end
end

@recipe function f(game::StackedInterceptMG, s::StackedInterceptState, σ1::AbstractVector, σ2::AbstractVector)
    (;attackers, defender) = s
    attacker1, attacker2 = attackers
    _pol1 = reshape(σ1, (4,4))
    pol11 = vec(sum(_pol1, dims=2)) |> permutedims
    pol12 = vec(sum(_pol1, dims=1)) |> permutedims
    pol2 = σ2 |> permutedims
    @series begin
        c       --> :blue
        lw      --> 10
        alpha   --> pol11
        action_lines(attacker1)
    end
    @series begin
        c       --> :blue
        lw      --> 10
        alpha   --> pol12
        action_lines(attacker2)
    end
    @series begin
        c       --> :red
        lw      --> 10
        alpha   --> pol2
        action_lines(defender)
    end
    @series begin
        game, s
    end
end

@recipe f(game::StackedInterceptMG, s::StackedInterceptState, σ1::SparseCat, σ2::SparseCat) = game, s, σ1.probs, σ2.probs

end
