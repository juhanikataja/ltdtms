module treeint

export integrate, gettree, Node

mutable struct Node{T} # {{{
  isleaf::Bool
  point::UInt64
  children::Union{Array{Node{T},1}, Nothing}
  limits::Array{T,2}
  Node(leaf::Bool, limits::Array{T,2}) where {T<:Number} = begin
    new{T}(leaf, UInt64(0), nothing, limits)
  end
  Node(leaf, point, children, limits::Array{T,2}) where {T<:Number} = begin
    new{T}(leaf, point, children, limits)
  end
  Node() = begin
    new{T}(true, 0, false, nothing)
  end
end # }}}

function getlimits(pts::Array{T, 2}, eps) where T <: AbstractFloat # {{{
  limits = Array{T,2}(undef,size(pts,2),2)
  for n = 1:size(pts,2)
    limits[n,1] = minimum(pts[:,n])
    limits[n,2] = maximum(pts[:,n])
  end
  add = diff(limits, dims=2).*eps
  limits[:,1] = limits[:,1].-add
  limits[:,2] = limits[:,2].+add
  return limits
end # }}}

function sublimit(limits::Array{T,2}, c::Tuple{Int64,Int64}) where T<:Number # {{{
  centrum = [sum(limits[1,:])/2 sum(limits[2,:])/2]
  if c == (1,1)
    return [limits[1,1] centrum[1]; limits[2,1] centrum[2]]
  elseif c == (1,2)
    return [centrum[1] limits[1,2]; limits[2,1] centrum[2]]
  elseif c == (2,1)
    return [limits[1,1] centrum[1]; centrum[2] limits[2,2]]
  elseif c == (2,2)
    return [centrum[1] limits[1,2]; centrum[2] limits[2,2]]
  else
    error("unsupported quadtree coordinates")
    return limits
  end
end # }}}

function sublimit(l::Array{T,2}, c::Tuple{Int64, Int64, Int64}) where T<:Number # {{{
    # c is in {z,y,x} coordinate order
    cc = sum(l, dims=2)/2 # centrum
      if c == (1,1,1) # {{{
        return [
                l[1,1] cc[1]; 
                l[2,1] cc[2];
                l[3,1] cc[3]
               ]
      elseif c == (1,1,2)
        return [
                cc[1] l[1,2];
                l[2,1] cc[2];
                l[3,1] cc[3]
               ]
      elseif c == (1,2,1)
        return [
                l[1,1] cc[1]; 
                cc[2] l[2,2];
                l[3,1] cc[3]
               ]
      elseif c == (1,2,2)
        return [
                cc[1] l[1,2];
                cc[2] l[2,2];
                l[3,1] cc[3]
               ]
      elseif c == (2,1,1)
        return [
                l[1,1] cc[1]; 
                l[2,1] cc[2];
                cc[3] l[3,2]
               ]
      elseif c == (2,1,2)
        return [
                cc[1] l[1,2]; 
                l[2,1] cc[2];
                cc[3] l[3,2]
               ]
      elseif c == (2,2,1)
        return [
                l[1,1] cc[1]; 
                cc[2] l[2,2];
                cc[3] l[3,2]
               ]
      elseif c == (2,2,2)
        return [
                cc[1] l[1,2]; 
                cc[2] l[2,2];
                cc[3] l[3,2]
               ]
      else
        error("unsupported octree coordinates")
        return limits3
      end # }}}
    end # }}}

function inlimits(limits, k, pts) # {{{
  for m = 1:size(pts, 2)
    if !(limits[m,1] < pts[k,m] && pts[k,m] < limits[m,2])
      return false
    end
  end
  return true
end # }}}

function getquadrant(children, k, pts) # {{{
  if !(isnothing(children))
    for i=1:length(children)
      if inlimits(children[i].limits, k, pts)
        return i
      end
    end
  end
  return 0
end # }}}

function addpoint!(n::Node{T}, k, pts, depth; maxdepth=6) where T<:Number # {{{
  if n.isleaf
    if n.point == 0
      n.point = k
      return
    else
      if depth < maxdepth
        n.isleaf = false
        n.children = 
          if size(n.limits,1) == 2
            Array{Node{T},1}([
                              Node(true, sublimit(n.limits, (1, 1))),
                              Node(true, sublimit(n.limits, (1, 2))),
                              Node(true, sublimit(n.limits, (2, 1))),
                              Node(true, sublimit(n.limits, (2, 2))),
                             ])
          elseif size(n.limits,1) == 3
            Array{Node{T},1}([
                              Node(true, sublimit(n.limits, (1, 1, 1))),
                              Node(true, sublimit(n.limits, (1, 1, 2))),
                              Node(true, sublimit(n.limits, (1, 2, 1))),
                              Node(true, sublimit(n.limits, (1, 2, 2))),
                              Node(true, sublimit(n.limits, (2, 1, 1))),
                              Node(true, sublimit(n.limits, (2, 1, 2))),
                              Node(true, sublimit(n.limits, (2, 2, 1))),
                              Node(true, sublimit(n.limits, (2, 2, 2))),
                             ])
          end
        n_quadrant = getquadrant(n.children, n.point, pts)
        k_quadrant = getquadrant(n.children, k, pts)
        addpoint!(n.children[n_quadrant], n.point, pts, depth+1; maxdepth=maxdepth)
        n.point = 0
        addpoint!(n.children[k_quadrant], k, pts, depth+1; maxdepth=maxdepth)
      end
      return
    end
  else
    if depth < maxdepth
      k_quadrant = getquadrant(n.children, k, pts)
      addpoint!(n.children[k_quadrant], k, pts, depth+1; maxdepth=maxdepth)
    end
  end
end # }}}

function showleaves(n::Node, pts; acc=[0]) # {{{
  if n.isleaf == true && n.point != 0
    acc[1] = acc[1] + 1
    print("acc = $(acc[1]), n = $(n.point), p = $(pts[n.point, :])\n")
  elseif n.isleaf == false
    for child in n.children
      showleaves(child, pts, acc=acc)
    end
  end
end # }}}

function integratefun(n::Node, fun, T; N=2)  # {{{
  sampleval = fun(n.limits[:,1])
  acc = zeros(T, size(sampleval)...)
  gp = Array{Float64, 2}(undef, N, size(n.limits, 1))
  if n.isleaf == false
    for child in n.children
      acc = acc .+ integratefun(child, fun, T; N=N)
    end
    return acc
  else

    qp, qw = if N == 2
      [0.21132486540518708, 0.7886751345948129], [0.5, 0.5]
    elseif N == 1
      [0.5], [1]
    elseif N == 3
      # [0, 0.5, 1], [1/6, 4/6, 1/6]
      [0.1127016653792583, 0.5, 0.8872983346207417], [5/18, 8/18, 5/18]
    end
    #
    Delta = diff(n.limits, dims=2)
    h = prod(Delta)
    for k = 1:size(n.limits,1)
      gp[:,k] = qp.*Delta[k] .+ n.limits[k,1]
    end

    if size(n.limits,1) == 2 # quadtree
      for i=1:N, j=1:N
        acc = acc .+ fun([gp[i,1], gp[j,2]])*h*qw[i]*qw[j]
      end
    elseif size(n.limits,1) == 3 # octree
      for i=1:N, j=1:N, k=1:N
        acc = acc .+ fun([gp[i,1], gp[j,2], gp[k,3]])*h*qw[i]*qw[j]*qw[k]
      end
    end
    return acc
  end
end # }}}

function integrate(fun, samples::Array; maxdepth=5, T=Float64, pad=0.2, N=2) # {{{ #TODO(BUG): pad=0.5 often fails!
  limits = getlimits(samples, pad);
  tree = Node(true, zero(UInt64), nothing, limits)
  for i = 1:size(samples, 1)
    addpoint!(tree, i, samples, 0; maxdepth=maxdepth)
  end
  integratefun(tree, fun, T; N=N)
end # }}}

function integrate(fun, tree::Node{S}; T=Float64, N=2) where S<:Number # {{{
  integratefun(tree, fun, T; N=N) 
end # }}}

function gettree(samples;maxdepth, pad) # {{{
  limits = getlimits(samples, pad);
  tree = Node(true, zero(UInt64), nothing, limits)
  for i = 1:size(samples, 1)
    addpoint!(tree, i, samples, 0; maxdepth=maxdepth)
  end
  return tree
end # }}

end # module
