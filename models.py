import enum
from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Enum,
    Date,
    Float,
    text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from database import Base


# Enums
class RoleEnum(enum.Enum):
    Admin = "Admin"
    Editor = "Editor"
    Viewer = "Viewer"
    SuperAdmin = "SuperAdmin"


class StatusEnum(enum.Enum):
    active = "active"
    inactive = "inactive"


class GenderEnum(enum.Enum):
    male = "male"
    female = "female"
    other = "other"
    ratherNotSay = "ratherNotSay"


class TestEnum(enum.Enum):
    cbc = "cbc"


# Models
class User(Base):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )

    name = Column(String, nullable=False)
    phone_no = Column(String, nullable=False)
    employee_id = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), nullable=False)
    status = Column(Enum(StatusEnum), nullable=False, server_default=text("'active'"))

    reports = relationship("Report", back_populates="generated_by")
    assignments_created = relationship("TestAssignment", back_populates="assigned_by", foreign_keys="TestAssignment.assigned_by_user_id")
    assignments_received = relationship("TestAssignment", back_populates="assigned_to", foreign_keys="TestAssignment.assigned_to_user_id")


# TODO Chaniging for ma'am to be changed
class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )

    name = Column(String, nullable=False)
    uhid = Column(String, unique=True, nullable=False)
    dob = Column(Date, nullable=False)
    phone_no = Column(String, nullable=False)
    # TODO: Make Aadhar unique
    aadhar = Column(String, nullable=False)
    gender = Column(Enum(GenderEnum), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tests = relationship("Test", back_populates="patient")
    samples = relationship("Sample", back_populates="patient")
    reports = relationship("Report", back_populates="patient")


class Test(Base):
    __tablename__ = "tests"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    # For now, we are using a custom sample ID to identify tests
    custom_sample_id = Column(String, unique=True, nullable=False)
    condition = Column(String, nullable=False)

    test_name = Column(Enum(TestEnum), nullable=False)
    test_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="tests")
    samples = relationship("Sample", back_populates="test")
    report = relationship("Report", back_populates="test", uselist=False)
    blood_counts = relationship("CompleteBloodCount", back_populates="test")
    assignments = relationship("TestAssignment", back_populates="test")


class Sample(Base):
    __tablename__ = "samples"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False)
    isProcessed = Column(Boolean, nullable=False, default=False)

    sample_location = Column(String, nullable=False)
    sample_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="samples")
    test = relationship("Test", back_populates="samples")
    cbc = relationship("CompleteBloodCount", back_populates="sample", uselist=False)
    annotations = relationship("Annotation", back_populates="sample")


class Report(Base):
    __tablename__ = "reports"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False)

    diagnosis = Column(String, nullable=False)
    report_location = Column(String, nullable=False)
    report_datetime = Column(DateTime(timezone=True), server_default=func.now())
    generated_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    patient = relationship("Patient", back_populates="reports")
    test = relationship("Test", back_populates="report")
    generated_by = relationship("User", back_populates="reports")


class CompleteBloodCount(Base):
    __tablename__ = "complete_blood_counts"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    sample_id = Column(
        UUID(as_uuid=True), ForeignKey("samples.id"), unique=True, nullable=False
    )
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False)

    haemoglobin = Column(Float)
    pcv = Column(Float)
    rbc_count = Column(Float)
    mcv = Column(Float)
    mch = Column(Float)
    mchc = Column(Float)
    rdw_cv = Column(Float)
    rdw_sd = Column(Float)
    platelet_count = Column(Float)
    tlc = Column(Float)
    neutrophil = Column(Float)
    lymphocyte = Column(Float)
    eosinophil = Column(Float)
    monocyte = Column(Float)
    basophil = Column(Float)
    abs_neutrophil_count = Column(Float)
    abs_monocyte_count = Column(Float)
    abs_lymphocyte_count = Column(Float)
    abs_eosinophil_count = Column(Float)
    abs_basophil_count = Column(Float)
    mpv = Column(Float)
    pdw = Column(Float)
    neutrophil_lymphocyte = Column(Float)
    lymphocyte_monocyte = Column(Float)
    pct = Column(Float)
    p_lcc = Column(Float)
    p_lcr = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sample = relationship("Sample", back_populates="cbc")
    test = relationship("Test", back_populates="blood_counts")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id"), nullable=False)
    source = Column(String, nullable=False)
    type = Column(String, nullable=False)
    tool = Column(String, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    isProcessed = Column(Boolean, nullable=False, default=False)
    isClassDetected = Column(Boolean, nullable=False, default=False)
    isSubClassDetected = Column(Boolean, nullable=False, default=False)
    isSegmented = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sample = relationship("Sample", back_populates="annotations")
    annotation_boxes = relationship("AnnotationBox", back_populates="annotation")


class AnnotationBox(Base):
    __tablename__ = "annotation_boxes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    annotation_id = Column(String, ForeignKey("annotations.id"), nullable=False)
    boxes = Column(JSON, nullable=False)
    class_name = Column(String, nullable=True)
    class_confidence = Column(Float, nullable=True)
    sub_class = Column(String, nullable=True)
    sub_class_confidence = Column(Float, nullable=True)
    crop_path = Column(String, nullable=True)
    segmentation_polygon = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation = relationship("Annotation", back_populates="annotation_boxes")
    wbc_sub_class_confidences = relationship("WBCSubClassConfidences", back_populates="annotation_box", uselist=False)


class WBCSubClassConfidences(Base):
    __tablename__ = "wbc_sub_class_confidences"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    annotation_box_id = Column(
        UUID(as_uuid=True), ForeignKey("annotation_boxes.id"), unique=True, nullable=False
    )
    neutrophil_confidence = Column(Float, nullable=True)
    lymphocyte_confidence = Column(Float, nullable=True)
    eosinophil_confidence = Column(Float, nullable=True)
    monocyte_confidence = Column(Float, nullable=True)
    basophil_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation_box = relationship("AnnotationBox", back_populates="wbc_sub_class_confidences")


class CellClasses(Base):
    __tablename__ = "cell_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name = Column(String, nullable=False, unique=True)
    colors = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sub_classes = relationship("CellSubClasses", back_populates="cell_class")


class CellSubClasses(Base):
    __tablename__ = "cell_sub_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    class_id = Column(UUID(as_uuid=True), ForeignKey("cell_classes.id"), nullable=False)
    sub_class_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cell_class = relationship("CellClasses", back_populates="sub_classes")


class TestAssignment(Base):
    __tablename__ = "test_assignments"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    assigned_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    assigned_to_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())

    assigned_by = relationship("User", back_populates="assignments_created", foreign_keys=[assigned_by_user_id])
    assigned_to = relationship("User", back_populates="assignments_received", foreign_keys=[assigned_to_user_id])
    test = relationship("Test", back_populates="assignments")
