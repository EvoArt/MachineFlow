using Turing, Distributions, MCMCChains, KernelDensity,AdvancedHMC
using StatsBase, CSV, DataFrames, Plots, StatsPlots
using ReverseDiff
using Combinatorics
Turing.setadbackend(:reversediff)
y5 = CSV.read("5y.csv")[1:end,3:end]# actualy values
names!(y5,[:aa,:od,:pc,:sr,:vg])
pred5 = CSV.read("5pred.csv")[1:end,3:end] # y value predictions (probabilities) from neural net
names!(pred5,[:aa,:od,:pc,:sr,:vg])

n = size(y5)[1] #number of observations

# Now I want to know the probability of guessing aa
# Given that the true value is aa

# Split the data in half. So, calculate probabilities
# from one half. And use them to model the other.
train_pred5 = pred5[1:n÷2,1:end]
train_y5 = y5[1:n÷2,1:end]
test_pred5 = pred5[n÷2:end,1:end]
test_y5= y5[n÷2:end,1:end]

function top(x,n = 3)
top3 = []
conf = []
tops = collect(permutations([1,2,3,4,5],n))
for i in 1:size(x)[1]
    row = Array(x[i,1:end])
    
    push!(top3,sortperm(row, rev = true)[1:n])
    push!(conf,maximum(row))
end

    top3s = [sum([j == k for j in top3]) for k in tops] ./ length(top3)
        return top3s
end



choices = train_pred5[train_y5[:aa] .== 1,1:end]
aatop3 = top(choices,3)
choices = train_pred5[train_y5[:od] .== 1,1:end]
odtop3 = top(choices,3)
choices = train_pred5[train_y5[:pc] .== 1,1:end]
pctop3 = top(choices,3)
choices = train_pred5[train_y5[:sr] .== 1,1:end]
srtop3 = top(choices,3)
choices = train_pred5[train_y5[:vg] .== 1,1:end]
vgtop3 = top(choices,3)


function y_top(x,n = 3)
    top3 = []
    conf = []
    tops = collect(permutations([1,2,3,4,5],n))
    for i in 1:size(x)[1]
        row = Array(x[i,1:end])
        
        push!(top3,sortperm(row, rev = true)[1:n])
        push!(conf,maximum(row))
    end
    
        top3s = [sum([j == k for j in top3]) for k in tops]
            return top3s
    end





@model top3_model(a,o,p,s,v, y, n) = begin
    #priors
   
    props ~ Dirichlet(5,1)
    
    # num is the number of aa in the community.


    #x = sum(rand(Uniform(0,1),num) .< right) + sum(rand(Uniform(0,1),n-num) .< wrong)
    #x = ((num .* right) .+ (wrong .* (n-num))) ./n
    x = (a .* props[1] .* n) .+ (o .* props[2] .* n) .+ (p .* props[3] .* n) .+ (s .* props[4] .* n) .+ (v .* props[5] .* n)
    #x1 = sample(1:4, Weights(Array(right)),num)

    #end
   
    y .~ Poisson.(x)
end;

#r = sample(1:size(test_y5)[1],Weights(1.0 .- Array(test_y5[1:end,1]) ), n÷4)
r = sample(1:size(test_y5)[1], n÷4)
pred_sub = test_pred5[r,1:end]
y_sub5 = test_y5[r,1:end]
y_vals1 = y_top(pred_sub,1)
y_vals2 = y_top(pred_sub,2)
y_vals3 = y_top(pred_sub,3)


top_chain1 = sample(top3_model(aatop,odtop,pctop,srtop,vgtop, y_vals1, n÷4), MH(), 1000000)
top_chain2 = sample(top3_model(aatop,odtop,pctop,srtop,vgtop, y_vals2, n÷4), MH(), 1000000)
top_chain3 = sample(top3_model(aatop,odtop,pctop,srtop,vgtop, y_vals3, n÷4), MH(), 1000000)

density(top_chain1["props[1]"], label = "aa1")
density!(top_chain2["props[1]"], label = "aa2")
density!(top_chain3["props[1]"], label = "aa3")
vline!([sum(y_sub5[:aa])/(n÷4)], label = "aa")

density(top_chain1["props[2]"], label = "od1")
density!(top_chain2["props[2]"], label = "od2")
density!(top_chain3["props[2]"], label = "od3")
vline!([sum(y_sub5[:od])/(n÷4)], label = "od")

density(top_chain1["props[3]"], label = "pc1")
density!(top_chain2["props[3]"], label = "pc2")
density!(top_chain3["props[3]"], label = "pc3")
vline!([sum(y_sub5[:pc])/(n÷4)], label = "pc")

density(top_chain1["props[4]"], label = "sr1")
density!(top_chain2["props[4]"], label = "sr2")
density!(top_chain3["props[4]"], label = "sr3")
vline!([sum(y_sub5[:sr])/(n÷4)], label = "sr")

density(top_chain1["props[5]"], label = "vg1")
density(top_chain2["props[5]"], label = "vg2")
density(top_chain3["props[5]"], label = "vg3")
vline!([sum(y_sub5[:vg])/(n÷4)], label = "vg")





