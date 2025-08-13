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
    dim.nzb = dim(nzb)
    if(is.null(dim.nzb))dim.nzb <- c(1,length(nzb))# singleton in x
    nzb = array(nzb > 0,dim.nzb)
    if(logical)nzb
    else apply(nzb,2,function(L,groupid)if(any(L))groupid[L]else NULL,groupid=seq(along.with=group))
    }

unstanCoef <- function(stan,intercepts,betas,df,lambda,K){
###  Check stan to see if standardization was done. If so, converts the coefficients
    if(!is.null(stan)){
        sc=stan$scales
### Now we make a matrix and use the adelie multiply to get the intercepts

        xzero=matrix.dense(matrix(0,1,length(sc)),method="naive")
        xzero = matrix.standardize(xzero,centers=stan$centers,scales=sc)
        if(K==1){
            preds = xzero$sp_tmul(betas)
            intercepts <- intercepts + t(preds)
        }
        else{ #K>1
            xzero = matrix.kronecker_eye(xzero,K=K)
            preds = xzero$sp_tmul(betas)
            nlams=ncol(preds)
            intercepts = matrix(intercepts,nlams,K)
            intercepts=intercepts+t(preds)
            sc=rep(sc,rep(K,length(sc)))
        }
        betas <- betas%*%Diagonal(x=1/sc)
        }
        list(intercepts=intercepts,betas=betas,df=df,lambda=lambda)
}

