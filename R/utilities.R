error.bars <-
function(x, upper, lower, width = 0.02, ...)
{
	xlim <- range(x)
	barw <- diff(xlim) * width
	segments(x, upper, x, lower, ...)
	segments(x - barw, upper, x + barw, upper, ...)
	segments(x - barw, lower, x + barw, lower, ...)
	range(upper, lower)
}
nonzeroGroup <- function(coefob,group, logical=FALSE){
    ## computes which groups are active along the path
    ## groups are numbered seq(along.with=group)
    K=ncol(coefob$intercepts)
    ## if K>1 we change the group def
    group = K*(group-1) +1
    nzb=coefob$betas
    nzb=as.matrix(nzb)
    nzb=nzb!=0
    ncols=ncol(nzb)
    nlams = nrow(nzb)
    nzb = matrix(apply(nzb,1,cumsum),ncols,nlams)#transposes
    nzb= rbind(rbind(0,nzb)[group,,drop=FALSE],nzb[ncols,,drop=FALSE])
    nzb = apply(nzb,2,diff)
    nzb = array(nzb > 0,dim(nzb))
    if(logical)nzb
    else apply(nzb,2,function(L,groupid)if(any(L))groupid[L]else NULL,groupid=seq(along.with=group))
    }
