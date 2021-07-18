"""
$(DocStringExtensions.README)
"""
module NeuralPDE

using DocStringExtensions
using Reexport, Statistics
@reexport using DiffEqBase

using Flux, Zygote, DiffEqSensitivity, ForwardDiff, Random, Distributions
using DiffEqFlux, Adapt, DiffEqNoiseProcess, CUDA, StochasticDiffEq
using ModelingToolkit
using GalacticOptim
using Quadrature,Cubature
using QuasiMonteCarlo
using RuntimeGeneratedFunctions
using SciMLBase
using Statistics
import Tracker, Optim
using DomainSets
using Symbolics
import ModelingToolkit: value, nameof, toexpr, build_expr, expand_derivatives
import DomainSets: Domain, ClosedInterval
import ModelingToolkit: Interval, infimum, supremum #,Ball
import SciMLBase: @add_kwonly
using Flux: @nograd
abstract type NeuralPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
"""
    TerminalPDEProblem(g, f, μ, σ, x0, tspan)
A semilinear parabolic PDE problem with a terminal condition.
Consider `du/dt = l(u) + f(u)`; where l is the nonlinear Lipschitz function
# Arguments
* `g` : The terminal condition for the equation.
* `f` : The function f(u)
* `μ` : The drift function of X from Ito's Lemma
* `μ` : The noise function of X from Ito's Lemma
* `x0`: The initial X for the problem.
* `tspan`: The timespan of the problem.
"""
struct TerminalPDEProblem{G,F,Mu,Sigma,X,T,P,A,UD,K} <: DiffEqBase.DEProblem
    g::G
    f::F
    μ::Mu
    σ::Sigma
    X0::X
    tspan::Tuple{T,T}
    p::P
    A::A
    u_domain::UD
    kwargs::K
    TerminalPDEProblem(g,f,μ,σ,X0,tspan,p=nothing;A=nothing,u_domain=nothing,kwargs...) = new{typeof(g),typeof(f),
                                                         typeof(μ),typeof(σ),
                                                         typeof(X0),eltype(tspan),
                                                         typeof(p),typeof(A),typeof(u_domain),typeof(kwargs)}(
                                                         g,f,μ,σ,X0,tspan,p,A,u_domain,kwargs)
end

Base.summary(prob::TerminalPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::TerminalPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

"""
    KolmogorovPDEProblem(f , g, phi , xspan , tspan, d, noise_rate_prototype = none)
A standard Kolmogorov PDE Problem.
# Arguments
* `f` : The drift function from Feynman-Kac representation of the PDE.
* `g` : The noise function from Feynman-Kac representation of the PDE.
* `phi` : The terminal condition for the PDE.
* `tspan`: The timespan for the problem.
* `xspan`: The xspan for the problem.
* `d`: The dimensions of the input x.
* `noise_rate_prototype`: A prototype type instance for the noise rates, that is the output g.
"""
struct KolmogorovPDEProblem{ F, G, Phi, X , T , D ,P,U0, ND} <: DiffEqBase.DEProblem
    f::F
    g::G
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    d::D
    p::P
    u0::U0
    noise_rate_prototype::ND
    KolmogorovPDEProblem( f, g, phi , xspan , tspan , d, p=nothing, u0=0 , noise_rate_prototype= nothing) = new{typeof(f),typeof(g),typeof(phi),eltype(tspan),eltype(xspan),typeof(d),typeof(p),typeof(u0),typeof(noise_rate_prototype)}(f,g,phi,xspan,tspan,d,p,u0,noise_rate_prototype)
end

Base.summary(prob::KolmogorovPDEProblem) = string(nameof(typeof(prob)))
function Base.show(io::IO, A::KolmogorovPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  print(io,"xspan: ")
  show(io,A.xspan)
  println(io , "μ")
  show(io , A.f)
  println(io,"Sigma")
  show(io , A.g)
end

abstract type ParametersDomain end
struct KolmogorovParamDomain{T} <:ParametersDomain
           sigma::Tuple{T,T}
           mu::Tuple{T,T}
           phi::Tuple{T,T}
end

struct ParamKolmogorovPDEProblem{ F, G, Phi, X , T , D, YD , P,U0 , YSP , YMP ,YPH, NP} <: DiffEqBase.DEProblem
    f::F
    g::G
    phi::Phi
    xspan::Tuple{X,X}
    tspan::Tuple{T,T}
    d::D
    Y_domain::YD
    p::P
    u0::U0
    Y_sigma_prototype::YSP
    Y_mu_prototype::YMP
    Y_phi_prototype::YPH
    noise_rate_prototype::NP
    ParamKolmogorovPDEProblem( f, g, phi , xspan , tspan , d, Y_domain, p=nothing, u0=0 ; Y_sigma_prototype=nothing , Y_mu_prototype=nothing , Y_phi_prototype=nothing , noise_rate_prototype= nothing) = new{typeof(f),typeof(g),typeof(phi),eltype(tspan),eltype(xspan),typeof(d),typeof(Y_domain),typeof(p),typeof(u0),typeof(Y_sigma_prototype),typeof(Y_mu_prototype),typeof(Y_phi_prototype),typeof(noise_rate_prototype)}(f,g,phi,xspan,tspan,d,Y_domain,p,u0 , Y_sigma_prototype,Y_mu_prototype,Y_phi_prototype,noise_rate_prototype)
end

Base.summary(prob::ParamKolmogorovPDEProblem) = string(nameof(typeof(prob)))
function Base.show(io::IO, A::ParamKolmogorovPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
  print(io,"xspan: ")
  show(io,A.xspan)
  println(io , "μ")
  show(io , A.f)
  println(io,"Sigma")
  show(io , A.g)
end

"""
Algorithm for solving differential equation using a neural network.

Arguments:
* `chain`: A Chain neural network
* `opt`: The optimizer to train the neural network. Defaults to `BFGS()`.
* `initθ`: The initial parameter of the neural network.
* `autodiff`: The switch between automatic and numerical differentiation for
              the PDE operators. The reverse mode of the loss function is always AD.
"""
struct NNODE{C,O,P,K} <: NeuralPDEAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    kwargs::K
end
function NNODE(chain,opt=Optim.BFGS(),init_params = nothing;autodiff=false,kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNODE(chain,opt,initθ,autodiff,kwargs)
end


include("ode_solve.jl")
include("pde_solve.jl")
include("pde_solve_ns.jl")
include("kolmogorov_solve.jl")
include("rode_solve.jl")
include("stopping_solve.jl")
include("pinns_pde_solve.jl")
include("param_kolmogorov_solve.jl")

export NNODE, TerminalPDEProblem, NNPDEHan, NNPDENS, NNRODE,
       KolmogorovPDEProblem, NNKolmogorov, NNStopping,ParamKolmogorovPDEProblem,KolmogorovParamDomain, NNParamKolmogorov,
       PhysicsInformedNN, discretize,
       GridTraining, StochasticTraining, QuadratureTraining, QuasiRandomTraining
       build_loss_function, get_loss_function,
       generate_training_sets, get_variables, get_argument, get_bounds
       get_phi, get_numeric_derivative,
       build_symbolic_equation, build_symbolic_loss_function, symbolic_discretize

end # module
