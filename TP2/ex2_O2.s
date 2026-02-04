	.file	"ex2.c"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"x\342\220\243=\342\220\243%f,\342\220\243y\342\220\243=\342\220\243%f,\342\220\243time\342\220\243=\342\220\243%f\342\220\243s\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB39:
	.cfi_startproc
	endbr64
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	subq	$16, %rsp
	.cfi_def_cfa_offset 32
	call	clock@PLT
	movsd	.LC1(%rip), %xmm0
	pxor	%xmm1, %xmm1
	movq	%rax, %rbx
	movl	$100000000, %eax
	.p2align 4,,10
	.p2align 3
.L2:
	addsd	%xmm0, %xmm1
	addsd	%xmm0, %xmm1
	subl	$2, %eax
	jne	.L2
	movsd	%xmm1, 8(%rsp)
	call	clock@PLT
	movsd	8(%rsp), %xmm1
	pxor	%xmm2, %xmm2
	leaq	.LC3(%rip), %rsi
	subq	%rbx, %rax
	movl	$2, %edi
	cvtsi2sdq	%rax, %xmm2
	movapd	%xmm1, %xmm0
	movl	$3, %eax
	divsd	.LC2(%rip), %xmm2
	call	__printf_chk@PLT
	addq	$16, %rsp
	.cfi_def_cfa_offset 16
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE39:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	1374389535
	.long	1073028792
	.align 8
.LC2:
	.long	0
	.long	1093567616
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
