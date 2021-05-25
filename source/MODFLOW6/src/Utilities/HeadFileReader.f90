module HeadFileReaderModule

  use KindModule
  use SimModule, only: store_error, store_error_unit, ustop
  use ConstantsModule, only: LINELENGTH

  implicit none
  
  private
  public :: HeadFileReaderType
  
  type :: HeadFileReaderType
    
    integer(I4B) :: inunit
    character(len=16) :: text
    integer(I4B) :: nlay
    integer(I4B) :: kstp
    integer(I4B) :: kper
    integer(I4B) :: kstpnext
    integer(I4B) :: kpernext
    logical :: endoffile
    real(DP) :: delt
    real(DP) :: pertim
    real(DP) :: totim
    real(DP), dimension(:), allocatable :: head
  
  contains
  
    procedure :: initialize
    procedure :: read_record
    procedure :: finalize
  
  end type HeadFileReaderType
  
  contains
  
  subroutine initialize(this, iu, iout)
! ******************************************************************************
! initialize
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(HeadFileReaderType) :: this
    integer(I4B), intent(in) :: iu
    integer(I4B), intent(in) :: iout
    ! -- local
    integer(I4B) :: kstp_last, kper_last
    logical :: success
! ------------------------------------------------------------------------------
    this%inunit = iu
    this%endoffile = .false.
    this%nlay = 0
    !
    ! -- Read the first head data record to set kstp_last, kstp_last
    call this%read_record(success)
    kstp_last = this%kstp
    kper_last = this%kper
    rewind(this%inunit)
    !
    ! -- Determine number of records within a time step
    if (iout > 0) &
      write(iout, '(a)') &
        'Reading binary file to determine number of records per time step.'
    do
      call this%read_record(success, iout)
      if (.not. success) exit
      if (kstp_last /= this%kstp .or. kper_last /= this%kper) exit
      this%nlay = this%nlay + 1
    enddo
    rewind(this%inunit)
    if (iout > 0) &
    write(iout, '(a, i0, a)') 'Detected ', this%nlay, &
      ' unique records in binary file.'
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
    class(HeadFileReaderType) :: this
    logical, intent(out) :: success
    integer(I4B), intent(in), optional :: iout_opt
    ! -- local
    integer(I4B) :: iostat, iout
    integer(I4B) :: ncol, nrow, ilay
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
    success = .true.
    this%kstpnext = 0
    this%kpernext = 0
    read(this%inunit, iostat=iostat) this%kstp, this%kper, this%pertim, &
      this%totim, this%text, ncol, nrow, ilay
    if (iostat /= 0) then
      success = .false.
      if (iostat < 0) this%endoffile = .true.
      return
    endif
    !
    ! -- allocate head to proper size
    if (.not. allocated(this%head)) then
      allocate(this%head(ncol*nrow))
    else
      if (size(this%head) /= ncol*nrow) then
        deallocate(this%head)
        allocate(this%head(ncol*nrow))
      endif
    endif
    !
    ! -- read the head array
    read(this%inunit) this%head
    !
    ! -- look ahead to next kstp and kper, then backup if read successfully
    if (.not. this%endoffile) then
      read(this%inunit, iostat=iostat) this%kstpnext, this%kpernext
      if (iostat == 0) then
        call fseek_stream(this%inunit, -2 * I4B, 1, iostat)
      else if (iostat < 0) then
        this%endoffile = .true.
      endif
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
    class(HeadFileReaderType) :: this
! ------------------------------------------------------------------------------
    close(this%inunit)
    if(allocated(this%head)) deallocate(this%head)
    !
    ! -- return
    return
  end subroutine finalize
  
end module HeadFileReaderModule
