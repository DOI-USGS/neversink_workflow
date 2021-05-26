module BudgetFileReaderModule

  use KindModule
  use SimModule, only: store_error, store_error_unit, ustop
  use ConstantsModule, only: LINELENGTH

  implicit none
  
  private
  public :: BudgetFileReaderType
  
  type :: BudgetFileReaderType
    
    logical :: hasimeth1flowja = .false.
    integer(I4B) :: inunit
    integer(I4B) :: nbudterms
    integer(I4B) :: kstp
    integer(I4B) :: kper
    integer(I4B) :: kstpnext
    integer(I4B) :: kpernext
    logical :: endoffile
    character(len=16) :: budtxt
    character(len=16), dimension(:), allocatable :: budtxtarray
    integer(I4B) :: nval
    integer(I4B) :: idum1
    integer(I4B) :: idum2
    integer(I4B) :: imeth
    integer(I4B), dimension(:), allocatable :: imetharray
    real(DP) :: delt
    real(DP) :: pertim
    real(DP) :: totim
    character(len=16) :: srcmodelname
    character(len=16) :: srcpackagename
    integer(I4B) :: ndat
    integer(I4B) :: naux
    integer(I4B), dimension(:), allocatable :: nauxarray
    character(len=16), dimension(:), allocatable :: auxtxt
    character(len=16), dimension(:, :), allocatable :: auxtxtarray
    integer(I4B) :: nlist
    real(DP), dimension(:), allocatable :: flowja
    integer(I4B), dimension(:), allocatable :: nodesrc
    integer(I4B), dimension(:), allocatable :: nodedst
    real(DP), dimension(:), allocatable :: flow
    real(DP), dimension(:, :), allocatable :: auxvar
    character(len=16) :: dstmodelname
    character(len=16) :: dstpackagename
    character(len=16), dimension(:), allocatable :: dstpackagenamearray
  
  contains
  
    procedure :: initialize
    procedure :: read_record
    procedure :: finalize
  
  end type BudgetFileReaderType
  
  contains
  
  subroutine initialize(this, iu, iout, ncrbud)
! ******************************************************************************
! initialize
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(BudgetFileReaderType) :: this
    integer(I4B), intent(in) :: iu
    integer(I4B), intent(in) :: iout
    integer(I4B), intent(out) :: ncrbud
    ! -- local
    integer(I4B) :: ibudterm
    integer(I4B) :: kstp_last, kper_last
    integer(I4B) :: maxaux
    logical :: success
! ------------------------------------------------------------------------------
    this%inunit = iu
    this%endoffile = .false.
    this%nbudterms = 0
    ncrbud = 0
    maxaux = 0
    !
    ! -- Determine number of budget terms within a time step
    if (iout > 0) &
      write(iout, '(a)') &
        'Reading budget file to determine number of terms per time step.'
    !
    ! -- Read through the first set of data for time step 1 and stress period 1
    do
      call this%read_record(success)
      if (.not. success) exit
      this%nbudterms = this%nbudterms + 1
      if (this%naux > maxaux) maxaux = this%naux
      if (this%kstp /= this%kstpnext .or. this%kper /= this%kpernext) &
        exit
    end do
    kstp_last = this%kstp
    kper_last = this%kper
    allocate(this%budtxtarray(this%nbudterms))
    allocate(this%imetharray(this%nbudterms))
    allocate(this%dstpackagenamearray(this%nbudterms))
    allocate(this%nauxarray(this%nbudterms))
    allocate(this%auxtxtarray(maxaux, this%nbudterms))
    this%auxtxtarray(:, :) = ''
    rewind(this%inunit)
    !
    ! -- Now read through again and store budget text names
    do ibudterm = 1, this%nbudterms
      call this%read_record(success, iout)
      if (.not. success) exit
      this%budtxtarray(ibudterm) = this%budtxt
      this%imetharray(ibudterm) = this%imeth
      this%dstpackagenamearray(ibudterm) = this%dstpackagename
      this%nauxarray(ibudterm) = this%naux
      this%auxtxtarray(1:this%naux, ibudterm) = this%auxtxt(:)
      if (this%srcmodelname == this%dstmodelname) then
        if(allocated(this%nodesrc)) ncrbud = max(ncrbud, maxval(this%nodesrc))
      endif
    enddo
    rewind(this%inunit)
    if (iout > 0) &
    write(iout, '(a, i0, a)') 'Detected ', this%nbudterms, &
      ' unique flow terms in budget file.'
    !
    ! -- return
    return
  end subroutine initialize
  
  subroutine read_record(this, success, iout_opt)
! ******************************************************************************
! read_record
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use InputOutputModule, only: fseek_stream
    ! -- dummy
    class(BudgetFileReaderType) :: this
    logical, intent(out) :: success
    integer(I4B), intent(in), optional :: iout_opt
    ! -- local
    integer(I4B) :: i, n, iostat, iout
    character(len=LINELENGTH) :: errmsg
! ------------------------------------------------------------------------------
    !
    if (present(iout_opt)) then
      iout = iout_opt
    else
      iout = 0
    endif
    !
    this%kstp = 0
    this%kper = 0
    this%budtxt = ''
    this%nval = 0
    this%naux = 0
    this%idum1 = 0
    this%idum2 = 0
    this%srcmodelname = ''
    this%srcpackagename = ''
    this%dstmodelname = ''
    this%dstpackagename = ''
     
    success = .true.
    this%kstpnext = 0
    this%kpernext = 0
    read(this%inunit, iostat=iostat) this%kstp, this%kper, this%budtxt, &
      this%nval, this%idum1, this%idum2
    if (iostat /= 0) then
      success = .false.
      if (iostat < 0) this%endoffile = .true.
      return
    endif
    read(this%inunit) this%imeth, this%delt, this%pertim, this%totim
    if(this%imeth == 1) then
      if (trim(adjustl(this%budtxt)) == 'FLOW-JA-FACE') then
        if(allocated(this%flowja)) deallocate(this%flowja)
        allocate(this%flowja(this%nval))
        read(this%inunit) this%flowja
        this%hasimeth1flowja = .true.
      else
        this%nval = this%nval * this%idum1 * abs(this%idum2)
        if(allocated(this%flow)) deallocate(this%flow)
        allocate(this%flow(this%nval))
        if(allocated(this%nodesrc)) deallocate(this%nodesrc)
        allocate(this%nodesrc(this%nval))
        read(this%inunit) this%flow
        do i = 1, this%nval
          this%nodesrc(i) = i
        enddo
      endif
    elseif (this%imeth == 6) then
      ! -- method code 6
      read(this%inunit) this%srcmodelname
      read(this%inunit) this%srcpackagename
      read(this%inunit) this%dstmodelname
      read(this%inunit) this%dstpackagename
      read(this%inunit) this%ndat
      this%naux = this%ndat - 1
      if(allocated(this%auxtxt)) deallocate(this%auxtxt)
      allocate(this%auxtxt(this%naux))
      read(this%inunit) this%auxtxt
      read(this%inunit) this%nlist
      if(allocated(this%nodesrc)) deallocate(this%nodesrc)
      allocate(this%nodesrc(this%nlist))
      if(allocated(this%nodedst)) deallocate(this%nodedst)
      allocate(this%nodedst(this%nlist))
      if(allocated(this%flow)) deallocate(this%flow)
      allocate(this%flow(this%nlist))
      if(allocated(this%auxvar)) deallocate(this%auxvar)
      allocate(this%auxvar(this%naux, this%nlist))
      read(this%inunit) (this%nodesrc(n), this%nodedst(n), this%flow(n), &
        (this%auxvar(i, n), i = 1, this%naux), n = 1, this%nlist)
    else
      write(errmsg, '(a, a)') 'ERROR READING: ', trim(this%budtxt)
      call store_error(errmsg)
      write(errmsg, '(a, i0)') 'INVALID METHOD CODE DETECTED: ', this%imeth
      call store_error(errmsg)
      call store_error_unit(this%inunit)
      call ustop()
    endif
    if (iout > 0) then
      write(iout, '(1pg15.6, a, 1x, a)') this%totim, this%budtxt, &
        this%dstpackagename
    endif
    !
    ! -- look ahead to next kstp and kper, then backup if read successfully
    if (.not. this%endoffile) then
      read(this%inunit, iostat=iostat) this%kstpnext, this%kpernext
      if (iostat == 0) then
        call fseek_stream(this%inunit, -2 * I4B, 1, iostat)
      else if (iostat < 0) then
        this%endoffile = .true.
      end if
    endif
    !
    ! -- return
    return
  end subroutine read_record
  
  subroutine finalize(this)
! ******************************************************************************
! budgetdata_finalize
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    class(BudgetFileReaderType) :: this
! ------------------------------------------------------------------------------
    close(this%inunit)
    if(allocated(this%auxtxt)) deallocate(this%auxtxt)
    if(allocated(this%flowja)) deallocate(this%flowja)
    if(allocated(this%nodesrc)) deallocate(this%nodesrc)
    if(allocated(this%nodedst)) deallocate(this%nodedst)
    if(allocated(this%flow)) deallocate(this%flow)
    if(allocated(this%auxvar)) deallocate(this%auxvar)
    !
    ! -- return
    return
  end subroutine finalize
  
end module BudgetFileReaderModule
